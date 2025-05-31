!pip install ultralytics --no-deps
!pip install --upgrade nvidia-cuda-nvcc-cu12==12.5.40

root_dir = "/content"
runtime_dir = !pwd
if runtime_dir[0] != root_dir:
    raise Exception("Runtime folder does not match root folder.")
!rm -rf *
import locale
locale.getpreferredencoding = lambda: "UTF-8"

import torch
import logging

orig_torch_load = torch.load

def torch_wrapper(*args, **kwargs):
    kwargs['weights_only'] = False
    return orig_torch_load(*args, **kwargs)

torch.load = torch_wrapper

NODE_CLASS_MAPPINGS = {}
__all__ = ['NODE_CLASS_MAPPINGS']

import os
import tensorflow as tf
from tensorflow import keras
from ultralytics import YOLO
from keras import layers, models, optimizers
from keras.preprocessing import image_dataset_from_directory
from ultralytics import settings
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)
import matplotlib.pyplot as plt

settings.update({"wandb": False})

!curl -u leprieto:{SECRET} -o 00_pseudo_model.zip "https://cloud.leonprieto.com/remote.php/dav/files/leprieto/University/UEM/TFM/resources/jupyter/02_pseudo_model.zip"
!unzip -q 00_pseudo_model.zip -d 00_pseudo_model
!rm 00_pseudo_model.zip

!curl -u leprieto:{SECRET} -o 01_dataset.zip "https://cloud.leonprieto.com/remote.php/dav/files/leprieto/University/UEM/TFM/resources/00_clean_filtered_dataset/dataset.zip"
!unzip -q 01_dataset.zip -d 01_dataset
!rm 01_dataset.zip

import numpy as np
import tensorflow as tf
import torch
from tensorflow.keras import Model, layers
from ultralytics import YOLO

def _torch_conv_weights(conv: torch.nn.Conv2d):
    w = conv.weight.detach().cpu().numpy()
    w = np.transpose(w, (2, 3, 1, 0))  # (H, W, in_c/groups, out_c)
    return [w] + ([conv.bias.detach().cpu().numpy()] if conv.bias is not None else [])

def _torch_bn_weights(bn: torch.nn.BatchNorm2d):
    return [
        bn.weight.detach().cpu().numpy(),
        bn.bias.detach().cpu().numpy(),
        bn.running_mean.detach().cpu().numpy(),
        bn.running_var.detach().cpu().numpy(),
    ]

def conv_bn_silu(x, tblock, name, *, log=False):
    tconv: torch.nn.Conv2d = tblock.conv
    tbn: torch.nn.BatchNorm2d = tblock.bn

    pad_h, pad_w = tconv.padding
    if pad_h or pad_w:
        x = layers.ZeroPadding2D(padding=(pad_h, pad_w), name=f"{name}_pad")(x)
        keras_padding = "valid"
    else:
        keras_padding = "valid"

    conv = layers.Conv2D(
        filters=tconv.out_channels,
        kernel_size=tconv.kernel_size,
        strides=tconv.stride,
        padding=keras_padding,
        dilation_rate=tconv.dilation,
        groups=tconv.groups,
        use_bias=tconv.bias is not None,
        name=f"{name}_conv",
    )
    x = conv(x)

    bn = layers.BatchNormalization(
        epsilon=tbn.eps,
        momentum=1 - tbn.momentum,
        name=f"{name}_bn",
    )
    x = bn(x)

    x = layers.Activation("swish", name=f"{name}_silu")(x)

    conv.set_weights(_torch_conv_weights(tconv))
    bn.set_weights(_torch_bn_weights(tbn))

    if log:
        k = conv.kernel.shape
        print(f"{name:20s} | g={tconv.groups:<2d} bias={tconv.bias is not None} | stride={tconv.stride}  pad=({pad_h})  W→{k}")

    return x

def convert_c2f(x, tblock, idx, *, log=False):
    x = conv_bn_silu(x, tblock.cv1, f"c2f{idx}_cv1", log=log)

    half = x.shape[-1] // 2
    x1 = layers.Lambda(lambda z: z[..., :half], name=f"c2f{idx}_split1")(x)
    x2 = layers.Lambda(lambda z: z[..., half:], name=f"c2f{idx}_split2")(x)

    parts = [x1, x2]
    for j, bottleneck in enumerate(tblock.m):
        identity = x2
        x2 = conv_bn_silu(x2, bottleneck.cv1, f"c2f{idx}_b{j}cv1", log=log)
        x2 = conv_bn_silu(x2, bottleneck.cv2, f"c2f{idx}_b{j}cv2", log=log)
        if bottleneck.add:
            x2 = layers.Add(name=f"c2f{idx}_b{j}_add")([identity, x2])
        parts.append(x2)

    x = layers.Concatenate(name=f"c2f{idx}_concat")(parts)
    x = conv_bn_silu(x, tblock.cv2, f"c2f{idx}_cv2", log=log)

    x = layers.Lambda(lambda z: z, name=f"stage{idx}_out")(x)
    return x

def convert_sppf(x, tblock, idx, *, log=False):
    x = conv_bn_silu(x, tblock.cv1, f"sppf{idx}_cv1", log=log)

    p1 = layers.MaxPooling2D(5, 1, "same", name=f"sppf{idx}_p1")(x)
    p2 = layers.MaxPooling2D(5, 1, "same", name=f"sppf{idx}_p2")(p1)
    p3 = layers.MaxPooling2D(5, 1, "same", name=f"sppf{idx}_p3")(p2)
    x = layers.Concatenate(name=f"sppf{idx}_concat")([x, p1, p2, p3])

    x = conv_bn_silu(x, tblock.cv2, f"sppf{idx}_cv2", log=log)
    x = layers.Lambda(lambda z: z, name=f"stage{idx}_out")(x)
    return x

def build_keras_backbone(t_backbone, input_shape=(320, 320, 3), *, verbose=False):
    inputs = layers.Input(shape=input_shape)
    x = inputs

    for idx, layer in enumerate(t_backbone.children()):
        lname = layer._get_name()
        if lname == "Conv":
            x = conv_bn_silu(x, layer, f"conv{idx}", log=verbose)
            x = layers.Lambda(lambda z: z, name=f"stage{idx}_out")(x)
        elif lname == "C2f":
            x = convert_c2f(x, layer, idx, log=verbose)
        elif lname == "SPPF":
            x = convert_sppf(x, layer, idx, log=verbose)
        else:
            raise ValueError(f"Unsupported layer type: {lname}")

    return Model(inputs, x, name="keras_yolov8_backbone")

def _make_k_debug(model, n_stages):
    outs = [model.get_layer(f"stage{i}_out").output for i in range(n_stages)]
    return Model(model.input, outs)

# ---------------------------------------------------------------------------
# ── Checkers ───────────────────────────────────────────────────────────────-
# ---------------------------------------------------------------------------

def compare_backbones(t_backbone, k_backbone, shape=(320, 320, 3), n=5, seed=0, *, verbose=False):
    if verbose:
        return compare_backbones_layerwise(t_backbone, k_backbone, shape, n, seed)
    mx_abs = mx_rel = 0.0
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    for _ in range(n):
        x_np = rng.standard_normal((1, *shape), dtype=np.float32)
        with torch.no_grad():
            y_t = t_backbone(torch.from_numpy(x_np).permute(0, 3, 1, 2)).cpu().numpy()
            y_t = np.transpose(y_t, (0, 2, 3, 1))
        y_k = k_backbone(x_np, training=False).numpy()
        d = np.abs(y_k - y_t)
        r = d / (np.abs(y_t) + 1e-8)
        mx_abs = max(mx_abs, d.max())
        mx_rel = max(mx_rel, r.max())
    print(f"overall max |Δ| = {mx_abs:.3e}\tmax relΔ = {mx_rel:.3e}")
    return mx_abs, mx_rel


def compare_backbones_layerwise(t_backbone, k_backbone, shape=(320, 320, 3), n=1, seed=0):
    k_debug = _make_k_debug(k_backbone, len(list(t_backbone.children())))
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    x_np = rng.standard_normal((1, *shape), dtype=np.float32)
    with torch.no_grad():
        x_t = torch.from_numpy(x_np).permute(0, 3, 1, 2)
        t_outs = []
        for m in t_backbone.children():
            x_t = m(x_t)
            t_outs.append(x_t.cpu().numpy().transpose(0, 2, 3, 1))

    k_outs = k_debug(x_np, training=False)

    for idx, (t_o, k_o) in enumerate(zip(t_outs, k_outs)):
        d = np.abs(k_o - t_o)
        print(f"stage {idx:<2d} | max |Δ| {d.max():.3e}  mean |Δ| {d.mean():.3e}")

    return None

yolo = YOLO("/content/00_pseudo_model/weights/best.pt")
t_backbone = torch.nn.Sequential(*list(yolo.model.model.children())[:10]).eval()

k_backbone = build_keras_backbone(t_backbone, verbose=True)
k_backbone.save("yolov8_backbone.keras", overwrite=True)

compare_backbones(t_backbone, k_backbone, n=1, verbose=True)

k_backbone.summary()

def resize_and_pad(image, label):
    TARGET_SIZE = 320
    PADDING_COLOR = 114

    image = tf.cast(image, tf.float32)

    shape = tf.shape(image)
    h = shape[0]
    w = shape[1]

    scale = tf.minimum(TARGET_SIZE / tf.cast(h, tf.float32),
                       TARGET_SIZE / tf.cast(w, tf.float32))

    new_h = tf.cast(tf.round(tf.cast(h, tf.float32) * scale), tf.int32)
    new_w = tf.cast(tf.round(tf.cast(w, tf.float32) * scale), tf.int32)

    resized = tf.image.resize(image, [new_h, new_w], method="bilinear")

    pad_h = TARGET_SIZE - new_h
    pad_w = TARGET_SIZE - new_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    padded = tf.pad(resized,
                    paddings=[[pad_top, pad_bottom],
                              [pad_left, pad_right],
                              [0, 0]],
                    constant_values=PADDING_COLOR)

    padded.set_shape([TARGET_SIZE, TARGET_SIZE, 3])
    padded = padded / 255.0
    return padded, tf.reshape(label, [1])

def set_trainable_layers(model, start: int, end: int):
    """Enable training for specified layer range in the model."""
    for layer in model.layers[start:end]:
        layer.trainable = True

def compile_model(model, learning_rate: float = 1e-5):
    """Compile the model with Adam optimizer and binary crossentropy loss."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

def get_callbacks(filepath: str):
    """Create ModelCheckpoint callback to save best model weights."""
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True
        )
    ]

def train_model(model, train_dataset, val_dataset, class_weights, epochs, callbacks):
    """Train the model and return the training history."""
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks
    )
    return history

def evaluate_model(model, val_dataset, weight_file):
    """Load best weights and evaluate the model."""
    model.load_weights(weight_file)
    loss, accuracy = model.evaluate(val_dataset)
    print(f"\nValidation Accuracy: {accuracy*100:.2f}%")
    return loss, accuracy

def print_loss_curves(history):
    """Print training and validation loss curves."""
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    print(f"Loss: {train_loss}")
    print(f"Validation Loss: {val_loss}")

def evaluate_performance(model, test_dataset, weight_file):
  true_labels = []
  predictions = []
  for images, labels in test_dataset:
      true_labels.extend(labels.numpy())
      batch_pred = model.predict(images)
      predictions.extend((batch_pred > 0.5).astype(int).flatten())

  true_labels = np.array(true_labels)
  predictions = np.array(predictions)

  cm = confusion_matrix(true_labels, predictions, labels=[0, 1])
  cm_display = ConfusionMatrixDisplay(cm, display_labels=['def', 'ok'])
  report = classification_report(true_labels, predictions, target_names=['def', 'ok'])
  cm_display.plot(cmap=plt.cm.Blues)
  plt.savefig(f"confusion_matrix_{weight_file}.svg")
  plt.show()
  print(report)

IMG_SIZE = (300, 300)
BATCH_SIZE = 32
EPOCHS = 100
DATA_DIR = '01_dataset/dataset/casting_data/casting_data/train/'

train_dataset, val_dataset = image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="both",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

test_dataset = image_dataset_from_directory(
    '01_dataset/dataset/casting_data/casting_data/test/',
    seed=42,
    image_size=IMG_SIZE,
    batch_size=1,
    label_mode='binary'
)

train_dataset = train_dataset.unbatch().map(resize_and_pad, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.unbatch().map(resize_and_pad, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.unbatch().map(resize_and_pad, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(1).prefetch(tf.data.AUTOTUNE)

k_backbone.trainable = False  # Freeze all layers

x = k_backbone.get_layer("stage9_out").output

# Classification head
x = layers.Conv2D(64, kernel_size=3, activation='relu')(x)
x = layers.MaxPooling2D()(x)

x = layers.Conv2D(128, kernel_size=3, activation='relu')(x)
x = layers.MaxPooling2D()(x)

x = layers.Flatten()(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(1, activation='sigmoid')(x)

model = Model(inputs=k_backbone.input, outputs=output)
compile_model(model, 1e-3)

labels = np.concatenate([y for x, y in train_dataset], axis=0)
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(labels),
    y=labels.flatten()
)
class_weights = {i: w for i, w in enumerate(class_weights)}
weight_file = "model.weights.h5"
callbacks = get_callbacks(weight_file)
history = train_model(model, train_dataset, val_dataset, class_weights, EPOCHS, callbacks)
evaluate_model(model, val_dataset, weight_file)
print_loss_curves(history)
evaluate_performance(model, test_dataset, weight_file)

# Stages composing the backbone.
i = 0
for layer in model.layers:
  if "_out" in layer.name:
    print(f"Layer: {layer.name} Index: {i}")
  i += 1

# Fine Tuning: Stage 8-9.
weight_file = "model89.weights.h5"
set_trainable_layers(model, 101, 130)
compile_model(model)
callbacks = get_callbacks(weight_file)
history = train_model(model, train_dataset, val_dataset, class_weights, 30, callbacks)
evaluate_model(model, val_dataset, weight_file)
print_loss_curves(history)
evaluate_performance(model, test_dataset, weight_file)

# Fine Tuning: Stage 6-7.
weight_file = "model67.weights.h5"
set_trainable_layers(model, 68, 100)
compile_model(model)
callbacks = get_callbacks(weight_file)
history = train_model(model, train_dataset, val_dataset, class_weights, 30, callbacks)
evaluate_model(model, val_dataset, weight_file)
print_loss_curves(history)
evaluate_performance(model, test_dataset, weight_file)

# Fine Tuning: Stage 4-5.
weight_file = "model45.weights.h5"
set_trainable_layers(model, 35, 67)
compile_model(model)
callbacks = get_callbacks(weight_file)
history = train_model(model, train_dataset, val_dataset, class_weights, 30, callbacks)
evaluate_model(model, val_dataset, weight_file)
print_loss_curves(history)
evaluate_performance(model, test_dataset, weight_file)

# Fine Tuning: Stage 2-3.
weight_file = "model23.weights.h5"
set_trainable_layers(model, 11, 34)
compile_model(model)
callbacks = get_callbacks(weight_file)
history = train_model(model, train_dataset, val_dataset, class_weights, 30, callbacks)
evaluate_model(model, val_dataset, weight_file)
print_loss_curves(history)
evaluate_performance(model, test_dataset, weight_file)

# Fine Tuning: Stage 0-1.
weight_file = "model01.weights.h5"
set_trainable_layers(model, 0, 10)
compile_model(model)
callbacks = get_callbacks(weight_file)
history = train_model(model, train_dataset, val_dataset, class_weights, 30, callbacks)
evaluate_model(model, val_dataset, weight_file)
print_loss_curves(history)
evaluate_performance(model, test_dataset, weight_file)

!zip -r *