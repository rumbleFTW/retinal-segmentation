import os
import argparse
import json
from pathlib import Path
import traceback

import cv2
import torch
import numpy as np
from tqdm import tqdm

from models import *


class Metrics:
    @staticmethod
    def iou(pred_mask, true_mask):
        intersection = np.logical_and(pred_mask, true_mask)
        union = np.logical_or(pred_mask, true_mask)
        iou_score = np.sum(intersection) / np.sum(union)
        return iou_score

    @staticmethod
    def dice_coefficient(pred_mask, true_mask):
        intersection = np.logical_and(pred_mask, true_mask)
        dice = (2.0 * np.sum(intersection)) / (np.sum(pred_mask) + np.sum(true_mask))
        return dice

    @staticmethod
    def pixel_accuracy(pred_mask, true_mask):
        correct_pixels = np.sum(pred_mask == true_mask)
        total_pixels = pred_mask.size
        pixel_acc = correct_pixels / total_pixels
        return pixel_acc

    @staticmethod
    def precision(pred_mask, true_mask):
        true_positive = np.sum(np.logical_and(pred_mask, true_mask))
        false_positive = np.sum(np.logical_and(pred_mask, np.logical_not(true_mask)))

        precision = true_positive / (true_positive + false_positive)

        return precision

    @staticmethod
    def recall(pred_mask, true_mask):
        true_positive = np.sum(np.logical_and(pred_mask, true_mask))
        false_negative = np.sum(np.logical_and(np.logical_not(pred_mask), true_mask))

        recall = true_positive / (true_positive + false_negative)

        return recall

    @staticmethod
    def f1_score(pred_mask, true_mask):
        precision = Metrics.precision(pred_mask, true_mask)
        recall = Metrics.recall(pred_mask, true_mask)
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    @staticmethod
    def calc_all(pred_mask, true_mask):
        return {
            "IoU": Metrics.iou(pred_mask, true_mask),
            "Pixel accuracy": Metrics.pixel_accuracy(pred_mask, true_mask),
            "Dice coefficient": Metrics.dice_coefficient(pred_mask, true_mask),
            "Precision": Metrics.precision(pred_mask, true_mask),
            "Recall": Metrics.recall(pred_mask, true_mask),
            "F1 score": Metrics.f1_score(pred_mask, true_mask),
        }

    @staticmethod
    def test(model, data_dir, checkpt):
        model = model
        #model.load_state_dict(torch.load(checkpt)) //uncomment this line for CUDA testing
        model.load_state_dict(torch.load(checkpt, map_location=torch.device('cpu')))
        print(checkpt)
        iou_scores = []
        dice_scores = []
        pixel_accs = []
        precisions = []
        recalls = []
        f1_scores = []

        for impath in tqdm(os.listdir(os.path.join(data_dir, "images"))):
            image = cv2.imread(os.path.join(data_dir, "images", impath))
            image = cv2.resize(image, (512, 512))
            x = np.transpose(image, (2, 0, 1))
            x = x / 255.0
            x = np.expand_dims(x, axis=0)
            x = x.astype(np.float32)
            x = torch.from_numpy(x)

            pred = model(x).detach().numpy()[0][0]
            pred[pred > 0] = 255
            pred[pred < 0] = 0
            true = cv2.imread(
                os.path.join(data_dir, "annotations", impath).replace(
                    "images", "annotations"
                ),
                0,
            )
            true = cv2.resize(true, (512, 512))
            perfs = Metrics.calc_all(pred_mask=pred, true_mask=true)

            iou_scores.append(perfs["IoU"])
            dice_scores.append(perfs["Dice coefficient"])
            pixel_accs.append(perfs["Pixel accuracy"])
            precisions.append(perfs["Precision"])
            recalls.append(perfs["Recall"])
            f1_scores.append(perfs["F1 score"])

        return {
            "Mean IoU": np.mean(iou_scores),
            "Mean Pixel accuracy": np.mean(pixel_accs),
            "Mean Dice coefficient": np.mean(dice_scores),
            "Mean Precision": np.mean(precisions),
            "Mean Recall": np.mean(recalls),
            "Mean F1 score": np.mean(f1_scores),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to dataset")
    parser.add_argument(
        "--network",
        type=str,
        help="newtork type; available options: (att_unet, unet, seg_net)",
    )
    parser.add_argument("--checkpt", type=str, help="path to checkpoint .pth file")

    model_dict = {"att_unet": AttentionUNet(), "unet": UNet(), "seg_net": SegNet()}

    args = parser.parse_args()

    try:
        res = Metrics.test(
            model=model_dict[args.network], checkpt=args.checkpt, data_dir=args.data
        )
        res["Data"] = args.data
        res["Network"] = args.network

        count = 1
        path = os.path.join("runs", "test", f"exp{count}")
        while os.path.exists(path):
            path = os.path.join("runs", "test", f"exp{count}")
            count += 1
        Path(path).mkdir(parents=True, exist_ok=True)

        file_path = os.path.join(path, f"results.json")
        f = open(file_path, "w")
        with open(file_path, "w") as f:
            json.dump(res, f, indent=4)

    except Exception as e:
        print("Error:", e)
        traceback.print_exc()
        exit()
