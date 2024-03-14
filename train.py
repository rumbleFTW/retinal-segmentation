import torch
from torchvision.utils import make_grid
import cv2
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import os
from pathlib import Path
import argparse
from glob import glob

from tqdm import tqdm

from models import *
from losses import *


class Dataset(torch.utils.data.Dataset):
    def __init__(self, images_path, masks_path, size=(512, 512)):
        self.images_path = images_path
        self.masks_path = masks_path
        self.n_samples = len(images_path)
        self.size = size

    def __getitem__(self, index):
        """Reading image"""
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        image = cv2.resize(image, self.size)
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        image = torch.from_numpy(image)

        """ Reading mask """
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.size)
        mask = mask / 255.0
        mask = np.expand_dims(mask, axis=0)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)

        return image, mask

    def __len__(self):
        return self.n_samples


def train(model, loader, optimizer, loss_fn, device):
    epoch_loss = 0.0
    model.train()
    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss = epoch_loss / len(loader)
    return epoch_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to data folder")
    parser.add_argument(
        "--network",
        type=str,
        help="newtork type; available options: (att_unet, unet, seg_net)",
    )
    parser.add_argument("--device", type=str, default="cpu", help="device to use")

    args = parser.parse_args()

    try:
        data_x = sorted(glob(os.path.join(args.data, "images", "*")))
        data_y = sorted(glob(os.path.join(args.data, "annotations", "*")))
        print(f"Dataset Loaded from {args.data}; Size: {len(data_x)} images")

        model_dict = {"att_unet": AttentionUNet(), "unet": UNet(), "seg_net": SegNet()}
        model = model_dict[args.network]
        print(f"Training {args.network} on {args.device}")

        device = torch.device(args.device)

    except Exception as e:
        print("Error:", e)
        exit()

    H = 512
    W = 512
    size = (H, W)
    batch_size = 2
    num_epochs = 100
    lr = 1e-4

    dataset = Dataset(data_x, data_y)

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5)
    loss_fn = DiceBCELoss()

    best_train_loss = float("inf")

    count = 1
    path = os.path.join("runs", "train", f"exp{count}")
    while os.path.exists(path):
        path = os.path.join("runs", "train", f"exp{count}")
        count += 1

    Path(path).mkdir(parents=True, exist_ok=True)
    file_path = os.path.join(path, f"{args.network}-weights.pth")
    f = open(file_path, "w")
    f.close()
    global_step = 0
    writer = SummaryWriter(os.path.join("runs", "train", f"exp{count}"))
    for epoch in tqdm(range(num_epochs)):
        train_loss = train(model, data_loader, optimizer, loss_fn, device)
        writer.add_scalar("Training loss", train_loss, global_step=global_step)
        if train_loss < best_train_loss:
            real = next(iter(data_loader))[0]
            real = real.to(device, dtype=torch.float32)
            with torch.no_grad():
                generated = model(real)
            writer.add_image(
                f"Generated",
                make_grid(generated),
                global_step=global_step,
            )
            best_train_loss = train_loss
            torch.save(model.state_dict(), file_path)
        global_step += 1
