root_dir = "/content"
runtime_dir = !pwd
if runtime_dir[0] != root_dir:
    raise Exception("Runtime folder does not match root folder.")
!rm -rf *
import locale
locale.getpreferredencoding = lambda: "UTF-8"
!pip install roboflow
!pip install ultralytics==8.0.196

import torch
import logging

orig_torch_load = torch.load

def torch_wrapper(*args, **kwargs):
    kwargs['weights_only'] = False

    return orig_torch_load(*args, **kwargs)

torch.load = torch_wrapper

NODE_CLASS_MAPPINGS = {}
__all__ = ['NODE_CLASS_MAPPINGS']

from roboflow import Roboflow
from ultralytics import YOLO
import numpy as np
import os
import cv2

from ultralytics import settings
settings.update({"wandb": False})

rf = Roboflow(api_key="{SECRET}")
project = rf.workspace("tfm-ks6ji").project("disc-detection-v2-zqrln")
version = project.version(1)
dataset = version.download("yolov8")

!mv Disc-detection-v2-$version.version "00_labeled_dataset"

model = YOLO("yolov8n-seg.pt")
result = model.train(
    data = os.path.join(root_dir, "00_labeled_dataset", "data.yaml"),
    epochs = 100,
    imgsz = 300,
    batch = 32,
    name = "weakly_supervised_disk",
    device = "0",
)

def compute_iou(pred_mask, true_mask):
    intersection = (pred_mask & true_mask).float().sum((1, 2))
    union = (pred_mask | true_mask).float().sum((1, 2))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou

def compute_dice(pred_mask, true_mask):
    intersection = (pred_mask & true_mask).float().sum((1, 2))
    dice = (2. * intersection + 1e-6) / (pred_mask.float().sum((1, 2)) + true_mask.float().sum((1, 2)) + 1e-6)
    return dice

def polygon_to_mask(coords, height, width):
    coords = np.array(coords, dtype=np.float32)
    coords = coords.reshape((-1, 1, 2)) * [width, height]
    coords = coords.astype(np.int32)

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [coords], 1)
    return torch.tensor(mask, dtype=torch.bool)

n = 0
inference_time = 0
ious = []
dices = []
for root, dirs, files in os.walk("/content/00_labeled_dataset/test/images"):
  for file in files:
    path = os.path.join(root, file)
    result = model.predict(path)[0]
    inference_time += result.speed['inference']
    with open(os.path.join(root, file).replace("images", "labels").replace(".jpg", ".txt")) as f:
      true_label = f.read()
      true_label = true_label.split(" ")

    label = []
    for coords in result.masks.xyn[0]:
      label.append(coords[0])
      label.append(coords[1])
    del true_label[0]

    iou = compute_iou(polygon_to_mask(label, 320, 320).unsqueeze(0), polygon_to_mask(true_label,320,320).unsqueeze(0))
    dice = compute_dice(polygon_to_mask(label, 320, 320).unsqueeze(0), polygon_to_mask(true_label,320,320).unsqueeze(0))

    ious.append(iou.item())
    dices.append(dice.item())

print("IOU: {} | DICE: {}.".format(np.mean(ious), np.mean(dices)))
print("Inference time: {} ms.".format(inference_time/len(ious)))

project.version(dataset.version).deploy(model_type="yolov8", model_path=result.save_dir)