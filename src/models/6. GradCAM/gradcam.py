!rm -rf *
!curl -u leprieto:{SECRET} -o dataset.zip "https://cloud.leonprieto.com/remote.php/dav/files/leprieto/University/UEM/TFM/resources/00_clean_filtered_dataset/dataset.zip"
!unzip -q dataset.zip
!rm dataset.zip

import keras
import tensorflow as tf
from keras.applications import DenseNet121
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import Image, display
from keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)
from sklearn.utils.class_weight import compute_class_weight
import os

image_size = (300, 300)
batch_size = 32
train_ds, val_ds = keras.utils.image_dataset_from_directory(
    "dataset/casting_data/casting_data/train",
    validation_split=0.2,
    subset="both",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
    label_mode = 'categorical'
    )

train_ds = train_ds.map(lambda x, y: (keras.applications.densenet.preprocess_input(x), y))
val_ds = val_ds.map(lambda x, y: (keras.applications.densenet.preprocess_input(x), y))

basemodel = DenseNet121(weights = 'imagenet', include_top = False,
                        input_shape = (300, 300, 3), pooling = None)

# for layer in basemodel.layers:
#   layer.trainable = False

x = tf.keras.layers.Flatten()(basemodel.output)
x = tf.keras.layers.Dropout(0.7)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(16, activation = 'relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(2, activation = 'softmax')(x)
m = tf.keras.models.Model(inputs = basemodel.input, outputs = x)
m.compile(loss = 'binary_crossentropy',
          optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001),
         metrics = ['accuracy', tf.keras.metrics.Precision(name = 'precision'),
                   tf.keras.metrics.Recall(name = 'recall')])

m.summary()

hist = m.fit(train_ds, epochs = 100, batch_size = 32,
             validation_data = val_ds,
             callbacks = [
                tf.keras.callbacks.EarlyStopping(patience = 10, monitor = 'val_loss', mode = 'min', restore_best_weights = True),
                tf.keras.callbacks.ReduceLROnPlateau(patience = 6, monitor = 'val_loss', mode = 'min', factor = 0.1)
            ],
             )

test_dataset = image_dataset_from_directory(
    '/content/dataset/casting_data/casting_data/test',
    seed=42,
    image_size=(300,300),
    batch_size=1,
    label_mode='categorical'  # One-hot encoded labels
)

test_dataset = test_dataset.map(lambda x, y: (keras.applications.densenet.preprocess_input(x), y))

true_labels = []
predictions = []

for images, labels in test_dataset:
    true_labels.extend(labels.numpy())
    batch_pred = m.predict(images)
    predictions.extend(batch_pred)

true_labels = np.array(true_labels)
predictions = np.array(predictions)

true_classes = np.argmax(true_labels, axis=1)
pred_classes = np.argmax(predictions, axis=1)

print(np.bincount(true_classes))


cm = confusion_matrix(true_classes, pred_classes, labels=[0, 1])
cm_display = ConfusionMatrixDisplay(cm, display_labels=['def', 'ok'])
report = classification_report(true_classes, pred_classes, target_names=['def', 'ok'], digits=4)

cm_display.plot(cmap=plt.cm.Blues)
plt.savefig("confusion_matrix.svg")
plt.show()
print(report)

def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.utils.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.utils.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Load the original image
    img = keras.utils.load_img(img_path)
    img = keras.utils.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = mpl.colormaps["jet"]

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    display(Image(cam_path))

!mkdir out_gradcam
m.layers[-1].activation = None
for path, directories, files in os.walk("/content/dataset/casting_data/casting_data/test/def_front"):
  for file in files:
    img_path = os.path.join(path, file)
    img_array = keras.applications.densenet.preprocess_input(get_img_array(img_path, size=(300,300)))
    heatmap = make_gradcam_heatmap(img_array, m, "relu")
    save_and_display_gradcam(img_path, heatmap, cam_path="/content/out_gradcam/" + file)

!zip -r outgc.zip out_gradcam/