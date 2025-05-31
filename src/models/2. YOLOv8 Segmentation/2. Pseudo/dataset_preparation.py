# prepare environment
root_dir = "/content"
runtime_dir = !pwd
if runtime_dir[0] != root_dir:
    raise Exception("Runtime folder does not match root folder.")

import logging
!pip install ultralytics
!pip install roboflow
from ultralytics import YOLO

logger = logging.getLogger()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
import os

import torch
import logging

orig_torch_load = torch.load

def torch_wrapper(*args, **kwargs):
    kwargs['weights_only'] = False

    return orig_torch_load(*args, **kwargs)

torch.load = torch_wrapper

NODE_CLASS_MAPPINGS = {}
__all__ = ['NODE_CLASS_MAPPINGS']

# download clean dataset
!rm -rf *
!curl -u leprieto:{SECRET} -o dataset.zip "https://cloud.leonprieto.com/remote.php/dav/files/leprieto/University/UEM/TFM/resources/00_clean_filtered_dataset/dataset.zip"
!unzip dataset.zip -d "00_clean_dataset"
!rm dataset.zip

# download dataset from roboflow
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="{SECRET}")
project = rf.workspace("tfm-ks6ji").project("disc-detection-v2-zqrln")
version = project.version(1)
dataset = version.download("yolov8")

!mv $dataset.name-$dataset.version "01_labeled_dataset"

!curl -u leprieto:{SECRET} -o 02_pure_model.zip "https://cloud.leonprieto.com/remote.php/dav/files/leprieto/University/UEM/TFM/resources/jupyter/00_pure_model.zip"
!mkdir 02_pure_model
!unzip 02_pure_model.zip -d 02_pure_model
!rm 02_pure_model.zip
clean_model = YOLO(os.path.join(root_dir,"02_pure_model/weights/best.pt"))

import shutil
import os

target_folder = "03_merged_dataset"

if os.path.exists(target_folder):
    shutil.rmtree(target_folder)

shutil.copytree("01_labeled_dataset", target_folder)

def labeled(file):
  for labeled_root, dirs, labeled_files in os.walk(target_folder):
    for labeled_file in labeled_files:
      if file.replace(".", "_") in labeled_file.replace("cast_ok_cast_ok", "cast_ok").replace("cast_def_cast_def", "cast_def"):
        return True
  return False

dest_images = os.path.join(target_folder, "train", "images")
dest_labels = os.path.join(target_folder, "train", "labels")
n = 0
infered_files = 0
inference_time = 0
for root, dirs, files in os.walk("00_clean_dataset/dataset/casting_data"):
  for file in files:
    n += 1
    if not labeled(file):
      infered_files += 1
      result = clean_model.predict(os.path.join(root, file), conf = 0.9)[0]
      inference_time += result.speed['inference']
      if result.masks is None:
        logger.info("Total Work: {} Low confidence discard. Discarded item: {}.".format(n, file) )
        continue
      label = "0"
      for coords in result.masks.xyn[0]:
        label += f" {coords[0]} {coords[1]}"
      with open(os.path.join(dest_labels, file.replace(".jpeg", ".txt")), "w") as f:
        f.write(label)
      shutil.copy(os.path.join(root, file), dest_images)
      logger.info("Total Work: {} | Add pseudo label for {}".format(n, file))

    else:
      logging.info("Item already labeled, discarded item: {}.")
print("Inference time: {} ms.".format(inference_time))
print("Total work: {} | Infered files: {}.".format(n, infered_files))

print(n)

!roboflow import -w tfm-ks6ji -p disc-detection-pseudo-v2 03_merged_dataset/

# Commented out IPython magic to ensure Python compatibility.
# %cd 03_merged_dataset
!zip -r ../04_pseudo_dataset.zip . -r
# %cd $root_dir
!curl -u leprieto:{SECRET} -T 04_pseudo_dataset.zip "https://cloud.leonprieto.com/remote.php/dav/files/leprieto/University/UEM/TFM/resources/jupyter/01_pseudo_dataset.zip"