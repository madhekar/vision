import os
import ast
import cv2
import numpy as np
import json
import joblib
# import tensorflow as tf
# from tensorflow.keras.models import load_model
import torch
from torchvision import models, transforms
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
    
    model = torch.load(os.path.join(filter_model_path, filter_model_name), weights_only=True)
    #model = load_model(os.path.join(filter_model_path, filter_model_name))

    class_names = joblib.load(os.path.join(filter_model_path, filter_model_classes))

    inverted_classes = dict(zip(class_names.values(), class_names.keys()))
    print(f"---* {inverted_classes}: {image_size}")
    return model, inverted_classes, image_size_int

def prep_img_infer(img, m, ic, isz,pp):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m.to(device)

    input_tensor = pp(img)

    input_batch = input_tensor.unsqueeze(0)
    input_batch = input_batch.to(device)

    with torch.no_eval():
        out = m(input_batch)



def execute():

    img = "/home/madhekar/work/home-media-app/data/input-data/img/madhekar/2596441a-e02f-588c-8df4-dc66a133fc99/af23fb36-9e89-4b81-9990-020b02fe1056.jpg"

    model, inverted_classes, isz = init_filter_model()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(isz),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    prep_img_infer(img, model=model, class_names=inverted_classes, image_size=isz, preprocess=preprocess)