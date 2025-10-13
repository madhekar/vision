import os
import getpass
from utils.config_util import config
import streamlit as st
from utils.util import adddata_util as adu
from streamlit_tree_select import tree_select
from utils.util import model_util as mu
from utils.util import storage_stat as ss

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import os


def load_init_params():
    # Define directories
    
    (filter_model_path, 
    train_data_path,
    validation_data_path,
    filter_model_name,
    filter_model_classes,
    image_size,
    batch_size) = config.filer_config_load()
    return filter_model_path, train_data_path, validation_data_path, filter_model_name, filter_model_classes, image_size, batch_size

def train_filter_model(filter_model_path, train_dir, validation_dir, model_file, classes_file, image_size, batch_size):
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
    base_model = MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet"
    )

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

    model.save(os.path.join(filter_model_path,model_file))

    np.save(os.path.join(filter_model_path, classes_file), train_generator.class_indices)
    #return list(train_generator.class_indices.keys())


def execute():
    fmp, tdp, vdp, fmn, fmc,isz, bsz = load_init_params()

    

    # filter_model_path, train_data_path, validation_data_path, filter_model_name, filter_model_classes, image_size, batch_size