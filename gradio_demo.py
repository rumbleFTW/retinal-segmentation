import os

import cv2
import torch
import gradio as gr
import numpy as np
from PIL import Image
from models import *


def demo(img, Model, CUDA):
    model_dict = {
        "AttentionUNet": [
            AttentionUNet(),
            os.path.join(os.getcwd(), "checkpts", "att_unet-weights.pth"),
        ],
        "UNet": [
            UNet(),
            os.path.join(os.getcwd(), "checkpts", "unet-weights.pth"),
        ],
        "SegNet": [
            SegNet(),
            os.path.join(os.getcwd(), "checkpts", "seg_net-weights.pth"),
        ],
    }

    device = torch.device("cuda") if CUDA else torch.device("cpu")
    model_arch = model_dict[Model][0]
    checkpt = model_dict[Model][1]
    model_arch = model_arch.to(torch.device(device))
    model_arch.load_state_dict(torch.load(checkpt, map_location=device))

    image = cv2.resize(img, (512, 512))
    x = np.transpose(image, (2, 0, 1))
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    x = x.astype(np.float32)
    x = torch.from_numpy(x)
    x = x.to(device)

    pred = model_arch(x).cpu().detach().numpy()[0][0]
    pred[pred > 0] = 1
    pred[pred < 0] = 0

    return pred


iface = gr.Interface(
    fn=demo,
    inputs=[
        gr.Image(),
        gr.Dropdown(["UNet", "SegNet", "AttentionUNet"]),
        "checkbox",
    ],
    outputs="image",
)

iface.launch(share=True)
