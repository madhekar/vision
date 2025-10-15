import os
import ast
import cv2
import numpy as np
import json
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
from utils.config_util import config


def init_filter_model():
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

    sz = ast.literal_eval(image_size)
    image_size_int = (int(sz[0]), int(sz[1]))

    model = load_model(os.path.join(filter_model_path, filter_model_name))

    class_names = joblib.load(os.path.join(filter_model_path, filter_model_classes))    
    
    inverted_classes = dict(zip(class_names.values(), class_names.keys()))
    print(f'---* {inverted_classes}: {image_size}')
    return model,inverted_classes, image_size_int

def predict_image(image_path, model, class_names, image_size):
    print(f'in ---{image_path}: {image_size}: {class_names}')

    img = tf.keras.preprocessing.image.load_img(image_path, target_size=image_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    img_array = img_array / 255.0  # Rescale pixels

    img = Image.open(image_path)
    img = img.resize(image_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, 0)
    img_array = img_array / 255.0

    print(f'---*&* {img_array}')
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    print(f"Image: {image_path}")
    print(f"Predicted class: {predicted_class} with confidence {confidence:.2f}")
    return predicted_class

def execute():

    img = "/home/madhekar/work/home-media-app/data/input-data/img/madhekar/2596441a-e02f-588c-8df4-dc66a133fc99/af23fb36-9e89-4b81-9990-020b02fe1056.jpg"

    model, inverted_classes, isz = init_filter_model()

    predict_image(img, model=model, class_names=inverted_classes, image_size=isz)