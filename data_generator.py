import os

import cv2
import numpy as np
import sklearn.neighbors as nn
import torch
from torch.utils.data import Dataset, DataLoader

from config import img_rows, img_cols, nb_neighbors, q_ab


def get_soft_encoding(image_ab, nn_finder, nb_q):
    h, w = image_ab.shape[:2]
    a = np.ravel(image_ab[:, :, 0])
    b = np.ravel(image_ab[:, :, 1])
    ab = np.vstack((a, b)).T                                    # (n, 2)
    # Get the distance to and the idx of the nearest neighbors
    dist_neighb, idx_neigh = nn_finder.kneighbors(ab)           # (n, 5)
    # Smooth the weights with a gaussian kernel
    sigma_neighbor = 5
    wts = np.exp(-dist_neighb ** 2 / (2 * sigma_neighbor ** 2)) # (n, 5)
    wts = wts / np.sum(wts, axis=1)[:, np.newaxis]              # (n, 5)
    # format the target
    y = np.zeros((ab.shape[0], nb_q))                           # (n, 313)
    idx_pts = np.arange(ab.shape[0])[:, np.newaxis]             # (n, 1)
    y[idx_pts, idx_neigh] = wts # (n, 313) - (n, 1) broadcasting to (n, 5) - (n, 5)
    y = y.reshape(h, w, nb_q)
    return y


class ColorDataset(Dataset):
    def __init__(self, data_root):
        super().__init__()
        self.data_root = data_root
        self.file_list = sorted(os.listdir(self.data_root))

        # Load the array of quantized ab value
        self.nb_q = q_ab.shape[0]
        # Fit a NN to q_ab
        self.nn_finder = nn.NearestNeighbors(n_neighbors=nb_neighbors, algorithm='ball_tree').fit(q_ab)

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        out_img_rows, out_img_cols = img_rows // 4, img_cols // 4
        filename = os.path.join(self.data_root, self.file_list[index])
        bgr = cv2.imread(filename)
        bgr = cv2.resize(bgr, (img_rows, img_cols), cv2.INTER_CUBIC)
        gray = cv2.imread(filename, 0)
        gray = cv2.resize(gray, (img_rows, img_cols), cv2.INTER_CUBIC)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        x = gray / 255.
        x = x[:, :, None]

        out_lab = cv2.resize(lab, (out_img_rows, out_img_cols), cv2.INTER_CUBIC)
        # Before: 0..255
        # After: -128..127
        out_ab = out_lab[:, :, 1:].astype(np.int32) - 128

        y = get_soft_encoding(out_ab, self.nn_finder, self.nb_q)

        # if np.random.random_sample() > 0.5:
        #     x = np.fliplr(x)
        #     y = np.fliplr(y)

        x_tens = torch.Tensor(x.transpose(2, 0, 1))
        y_tens = torch.Tensor(y.transpose(2, 0, 1))
        return x_tens, y_tens


def create_dataloader(data_root, batch_size=16, shuffle=False):
    dataset = ColorDataset(data_root)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader