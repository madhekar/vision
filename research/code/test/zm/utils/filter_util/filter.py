import os
import ast
import numpy as np
import json
import joblib
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping
from utils.config_util import config

# Define directories, files, params
def load_init_params():
    (
        filter_model_path,
        train_data_path,
        validation_data_path,
        traing_data_path,
        trainging_map_file,
        filter_model_name,
        filter_model_classes,
        image_size,
        batch_size,
    ) = config.filer_config_load()

    batch_size_int = int(batch_size)
    sz = ast.literal_eval(image_size)
    image_size_int = (int(sz[0]), int(sz[1]))
    return filter_model_path, train_data_path, validation_data_path, traing_data_path, trainging_map_file, filter_model_name, filter_model_classes, image_size_int, batch_size_int

def train_filter_model(train_dir, validation_dir, model_file, classes_file, image_size, batch_size):

    print(f'{train_dir}:{validation_dir}:{model_file}:{classes_file}:{image_size}:{batch_size}')

    # Use ImageDataGenerator to load and augment images
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
    )

    # Load the pre-trained MobileNetV2 model (without the top classification layer)
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")

    # Freeze the base model layers to prevent re-training
    base_model.trainable = False

    # Add a new classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    predictions = Dense(3, activation="softmax")(
        x
    )  # 3 classes: document, scenic, people

    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(model)
    # Monitor 'val_loss' and wait for 5 epochs without improvement before stopping
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",  # Metric to monitor
        patience=5,  # Number of epochs with no improvement after which training will be stopped
        restore_best_weights=True,  # Restore model weights from the epoch with the best value of the monitored metric
        verbose=1,  # Print messages when early stopping is triggered
    )
    # Train the model
    history = model.fit(
        train_generator,
        epochs=100,
        validation_data=validation_generator,
        callbacks=[early_stopping_callback],
    )

    model.save(model_file)
    joblib.dump(train_generator.class_indices, classes_file)


# Make a prediction on a new image
def predict_image(image_path, model, class_names, image_size):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=image_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    img_array = img_array / 255.0 # Rescale pixels

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    print(f"Image: {image_path}")
    print(f"Predicted class: {predicted_class} with confidence {confidence:.2f}")
    return predicted_class

# Example usage (assuming you have a test image named 'test_image.jpg')
#class_names = list(train_generator.class_indices.keys())
def test_model(model, class_names, testing_path, Testing_map_file, image_size):
    y_src = []
    y_tar = []
    inverted_classes = dict(zip(class_names.values(), class_names.keys()))
    with open(os.path.join(testing_path, Testing_map_file)) as f:
        dlist = json.load(f)
        print('--->', dlist)
        for d in dlist:
            p_class = predict_image(os.path.join(testing_path,d["img"]), model, inverted_classes, image_size)
            y_src.append(d['label'])
            y_tar.append(p_class)
            print(f'predicted: {p_class} actual: {d["label"]}' )  
    print(classification_report(y_src, y_tar))

def execute():

    fmp, trdp, vdp, tsdp, tmf, fmn, fmc,isz, bsz = load_init_params()

    #train_filter_model(trdp, vdp, os.path.join(fmp, fmn), os.path.join(fmp, fmc), isz, bsz)

    model = load_model(os.path.join(fmp, fmn))

    class_names = joblib.load(os.path.join(fmp, fmc))

    print(class_names)
    
    test_model(model, class_names, tsdp, tmf, isz)
   
  