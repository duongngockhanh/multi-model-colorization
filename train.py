import os
import time

import wandb
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from data_generator import create_dataloader
from utils import categorical_crossentropy_color, merge_lab
from model import Zhang_Cla_Lab
from config import (train_root, val_root,
                    device, epochs, lr, 
                    train_batch_size, val_batch_size,
                    train_num_max, val_num_max, 
                    pretrained, saved_weight_path, 
                    use_wandb, wandb_proj_name, wandb_config)


def show_image_wandb(val_loader, model, val_batch_size, device, epoch):
    model.eval()
    with torch.no_grad():
        val_iter = iter(val_loader)
        l_inputs, ab_gts = next(val_iter)
        l_inputs, ab_gts = l_inputs.to(device), ab_gts.to(device)
        l_inputs *= 255
        ab_preds = model(l_inputs)
        img_preds = merge_lab(l_inputs, ab_preds)
        img_gts = merge_lab(l_inputs, ab_gts)
        
        images_pred = []
        images_gt = []

        fixed_num_showed_image = 5
        num_showed_image = val_batch_size if val_batch_size < fixed_num_showed_image else fixed_num_showed_image

        for i in range(num_showed_image):
            rgb_pred = cv2.cvtColor(img_preds[i], cv2.COLOR_LAB2RGB)
            pil_pred = Image.fromarray(rgb_pred)
            image_pred = wandb.Image(pil_pred, caption=f"epoch {epoch}")
            images_pred.append(image_pred)

            rgb_gt = cv2.cvtColor(img_gts[i], cv2.COLOR_LAB2RGB)
            pil_gt = Image.fromarray(rgb_gt)
            image_gt = wandb.Image(pil_gt, caption=f"epoch {epoch}")
            images_gt.append(image_gt)
    
    return images_pred, images_gt


def evaluate(model, dataloader, criterion, device, val_batch_size, val_num_max):
    model.eval()
    losses = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            if val_batch_size * len(losses) > val_num_max:
                break
    loss = sum(losses) / len(losses)
    return loss


def fit(model, train_loader, val_loader, saved_weight_path,
        criterion, optimizer, device, epochs, 
        train_batch_size, val_batch_size,
        train_num_max, val_num_max,
        use_wandb, wandb_proj_name, wandb_config
        ):

    if use_wandb == True:
        wandb.init(
            project=wandb_proj_name,
            config=wandb_config
        )
    
    train_losses = []
    val_losses = []

    best_val_loss = 10e5
    start_time = time.time()

    for epoch in range(epochs):
        start_time_epoch = time.time()
        batch_train_losses = []

        model.train()
        for _, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            batch_train_losses.append(loss.item())
            if train_batch_size * len(batch_train_losses) > train_num_max:
                break

        train_loss = sum(batch_train_losses) / len(batch_train_losses)
        train_losses.append(train_loss)

        val_loss = evaluate(model, val_loader, criterion, device, val_batch_size, val_num_max)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            torch.save(model.state_dict(), saved_weight_path)
            best_val_loss = val_loss

        # Show image
        if use_wandb == True:
            images_pred, images_gt = show_image_wandb(val_loader, model, val_batch_size, device, epoch)
            wandb.log({"train_loss": train_loss, "val_loss": val_loss, "images_pred": images_pred, "images_gt": images_gt})

        print(f'EPOCH {epoch + 1}:\tTrain loss: {train_loss:.4f}\tVal loss: {val_loss:.4f}\tTime: {time.time() - start_time_epoch:.2f}s')

    print(f"Complete training in {time.time() - start_time:2f}s")

    if use_wandb == True:
        wandb.finish()

    return train_losses, val_losses


def main():
    train_loader = create_dataloader(train_root, batch_size=train_batch_size, shuffle=True)
    val_loader = create_dataloader(val_root, batch_size=val_batch_size, shuffle=False)

    model = Zhang_Cla_Lab().to(device)
    criterion = categorical_crossentropy_color
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if pretrained != None:
        print(f"Load model from {pretrained}")
        model.load_state_dict(torch.load(pretrained))

    train_losses, val_losses = fit(
        model, train_loader, val_loader, saved_weight_path,
        criterion, optimizer, device, epochs,
        train_batch_size, val_batch_size,
        train_num_max, val_num_max,
        use_wandb, wandb_proj_name, wandb_config
    )

    print(f"Weights will be saved in {saved_weight_path}")

if __name__ == "__main__":
    main()