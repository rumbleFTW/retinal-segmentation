import os
from pathlib import Path
import cv2

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

from models import *


def evaluate(model, loader, loss_fn, device):
    epoch_loss = 0.0

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

        epoch_loss = epoch_loss / len(loader)
    return epoch_loss


def test(model, checkpt, impath, device="cpu"):
    model = model
    model = model.to(torch.device(device))
    model.load_state_dict(torch.load(checkpt, map_location=device))

    image = cv2.imread(impath)
    image = cv2.resize(image, (512, 512))
    x = np.transpose(image, (2, 0, 1))
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    x = x.astype(np.float32)
    x = torch.from_numpy(x)
    x = x.to(device)

    pred = model(x)
    count = 1
    path = os.path.join("runs", "test", f"exp{count}")
    while os.path.exists(path):
        path = os.path.join("runs", "test", f"exp{count}")
        count += 1
    Path(path).mkdir(parents=True, exist_ok=True)
    file_path = os.path.join(path, f"test-output.jpg")
    f = open(file_path, "w")
    f.close()
    plt.imsave(
        file_path,
        pred.detach().numpy()[0][0],
        cmap="gray",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, help="path to image to test")
    parser.add_argument(
        "--network",
        type=str,
        help="newtork type; available options: (att_unet, unet, seg_net, sem_unet)",
    )
    parser.add_argument("--device", type=str, default="cpu", help="device to use")
    parser.add_argument("--checkpt", type=str, help="path to checkpoint .pth file")

    model_dict = {"att_unet": AttentionUNet(), "unet": UNet(), "seg_net": SegNet()}

    args = parser.parse_args()

    try:
        test(
            model=model_dict[args.network],
            checkpt=args.checkpt,
            impath=args.img,
            device=args.device,
        )
    except Exception as e:
        print("Error:", e)
        exit()
