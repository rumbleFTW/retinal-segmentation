import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
import numpy as np

from models import UNet, SegNet, AttentionUNet


class Ensemble(nn.Module):
    def __init__(self, unet, segnet, att_unet) -> None:
        super(Ensemble, self).__init__()
        self.unet = unet
        self.segnet = segnet
        self.att_unet = att_unet

    def forward(self, X):
        Y = self.unet(X)
        Y_dashed = torch.zeros((1, 3, 512, 512), device="cpu")
        Y_reshaped = Y.unsqueeze(0).unsqueeze(0)
        Y_dashed[:, :, :, :] = Y_reshaped
        Y = self.segnet(Y_dashed)
        Y_reshaped = Y.unsqueeze(0).unsqueeze(0)
        Y_dashed[:, :, :, :] = Y_reshaped
        return self.att_unet(Y_dashed)


if __name__ == "__main__":
    unet = UNet().to("cpu")
    unet.load_state_dict(
        torch.load("./runs/train/exp1/unet-weights.pth", map_location="cpu")
    )
    segnet = SegNet().to("cpu")
    segnet.load_state_dict(
        torch.load("./runs/train/exp3/seg_net-weights.pth", map_location="cpu")
    )
    att_unet = AttentionUNet().to("cpu")
    att_unet.load_state_dict(
        torch.load("./runs/train/exp2/att_unet-weights.pth", map_location="cpu")
    )

    ens = Ensemble(unet=unet, att_unet=att_unet, segnet=segnet).to("cpu")

    image = cv2.imread("./data/DRIVE/images/01.png")
    image = cv2.resize(image, (512, 512))
    x = np.transpose(image, (2, 0, 1))
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    x = x.astype(np.float32)
    x = torch.from_numpy(x)
    x = x.to("cpu")

    pred = ens(x)

    plt.imsave(
        "./runs/test/pred.png",
        pred.detach().numpy()[0][0],
        cmap="gray",
    )
