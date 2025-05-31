!pip install ultralytics
!pip install scikit-learn

import os
import shutil
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

def split_train_val(base_dir, val_size=0.2):
    classes = [folder for folder in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, folder))]

    train_images = []
    val_images = []

    for class_name in classes:
        class_folder = os.path.join(base_dir, class_name)
        images = [os.path.join(class_folder, f) for f in os.listdir(class_folder) if f.endswith('.jpeg')]

        if len(images) == 0:
            continue

        train, val = train_test_split(images, test_size=val_size, random_state=42)
        train_images.extend([(img, class_name) for img in train])
        val_images.extend([(img, class_name) for img in val])

    if len(train_images) == 0:
        raise ValueError("No images were selected for the training set.")
    if len(val_images) == 0:
        raise ValueError("No images were selected for the validation set.")

    train_dir = "/content/split/train"
    val_dir = "/content/split/val"

    for class_name in classes:
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
    for img, class_name in train_images:
        shutil.move(img, os.path.join(train_dir, class_name, os.path.basename(img)))
    for img, class_name in val_images:
        shutil.move(img, os.path.join(val_dir, class_name, os.path.basename(img)))

    return train_dir, val_dir


def train_yolo(epochs=100, batch_size=32):
    model = YOLO("yolov8n-cls.pt")
    model.train(data="/content/split", epochs=epochs, batch=batch_size, imgsz = 300, device="0")
    return model

def test_and_generate_matrix(model, test_dir):
    results = model.predict(os.path.join(test_dir, "ok_front")) + model.predict(os.path.join(test_dir, "def_front"))

    y_true = []
    y_pred = []
    inference_time = 0
    i = 0
    for result in results:

      true_label = result.path.split('/')[-2]
      y_true.append(true_label)

      predicted_class = result.probs.top1
      predicted_label = model.names[predicted_class]
      y_pred.append(predicted_label)
      if true_label != predicted_label:
        result.show()
        result.save(f"missclassification_{i}.png")

      inference_time += result.speed['inference']
      i += 1

    inference_time = np.mean(inference_time)
    print(f"Inference time: {inference_time:.2f} ms")

    cm = confusion_matrix(y_true, y_pred, labels=['def_front', 'ok_front'])
    cm_display = ConfusionMatrixDisplay(cm, display_labels=model.names.values())

    cm_display.plot(cmap=plt.cm.Blues)
    plt.show()

    report = classification_report(y_true, y_pred, target_names=model.names.values())
    print(report)

!rm -rf *
!curl -u leprieto:${SECRET} -o dataset.zip "https://cloud.leonprieto.com/remote.php/dav/files/leprieto/University/UEM/TFM/resources/00_clean_filtered_dataset/dataset.zip"
!unzip -q dataset.zip
!rm dataset.zip

base_dir = '/content/dataset/casting_data/casting_data'

train_data_dir, val_data_dir = split_train_val(os.path.join(base_dir, 'train'))

model = train_yolo()

test_dir = '/content/dataset/casting_data/casting_data/test'
test_and_generate_matrix(model, test_dir)