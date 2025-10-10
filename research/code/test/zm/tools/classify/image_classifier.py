import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.utils import load_img, img_to_array
# --- 1. Load and prepare the data ---

# Define the training and validation data paths.
# For a real project, you would have separate folders for validation.
data_dir = os.path.join('/home/madhekar/work/home-media-app/models','img_classify/')

img_height, img_width = 224, 224
batch_size = 32

# Create TensorFlow datasets from the directory structure
train_ds = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="train",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validate",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Get the class names from the dataset
class_names = train_ds.class_names
print("Class names:", class_names)

# --- 2. Build the CNN model ---

num_classes = len(class_names)
model = keras.Sequential([
    # Rescale pixel values to a [0, 1] range
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),

    # Convolutional and pooling layers for feature extraction
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    # Flatten the features and add a dense layer for classification
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax') # Softmax for multi-class classification
])

# --- 3. Compile and train the model ---

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

model.summary()

epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# --- 4. Make a prediction on a new image ---



# Assume you have a new image named 'new_image.jpg' in the project directory
new_image_path = 'new_image.jpg'

img = load_img(new_image_path, target_size=(img_height, img_width))
img_array = img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

# To add a new image to your testing directory, use this line:
# from shutil import copyfile
# copyfile(new_image_path, f'data/{class_names[np.argmax(score)]}/new_image.jpg')

