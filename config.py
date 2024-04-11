import os

import scipy.io
import numpy as np
import torch

img_rows, img_cols = 256, 256
channel = 3
num_classes = 313
epsilon = 1e-8
epsilon_sqr = epsilon ** 2
nb_neighbors = 5
T = 0.38 # temperature parameter T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
q_ab = np.load("data/pts_in_hull.npy")
mat = scipy.io.loadmat('human_colormap.mat')
color_map = (mat['colormap'] * 256).astype(np.int32)

# Load the color prior factor that encourages rare colors
prior_factor = torch.from_numpy(np.load("data/prior_factor.npy")).to(device)
weights = prior_factor.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

# Hyperparameters
epochs = 100
lr = 1e-5
train_num_max = 4000
val_num_max = 400
pretrained = None
save_dir = "exp_Zhang_Cla_Lab"

train_root = "/kaggle/input/aio-coco-stuff/train2017/train2017"
val_root = "/kaggle/input/aio-coco-stuff/val2017/val2017"
train_batch_size = 32
val_batch_size = 8

# Save weight
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
saved_weights = sorted(os.listdir(save_dir))
if len(saved_weights) == 0:
    saved_weight_file = "exp01.pt"
    saved_weight_path = os.path.join(save_dir, saved_weight_file)
else:
    saved_weight_file = f"exp{int(saved_weights[-1][3:-3]) + 1:02d}.pt"
    saved_weight_path = os.path.join(save_dir, saved_weight_file)

# Use WanDB
use_wandb = True 
wandb_proj_name = "Zhang_Cla_Lab"
wandb_config = {
    "dataset": "coco-stuff",
    "model": "Zhang_Cla_Lab",
    "epochs": epochs,
    "lr": lr,
    "criterion": "categorical_crossentropy",
    "optimizer": "Adam",
    "train_num_max": train_num_max,
    "val_num_max": val_num_max,
    "pretrained": pretrained,
    "saved_weight_path": saved_weight_path
}
