import os
import cv2
import torch
import pyiqa
from PIL import Image
from torchvision import transforms

'''
import cv2 as cv
import os

"""
https://pyimagesearch.com/2020/04/06/blur-and-anonymize-faces-with-opencv-and-python/
"""

sample_path = "/home/madhekar/work/home-media-app/data/train-data/img/AnjaliBackup"

for rt, _, files in os.walk(sample_path, topdown=True):
    for file in files:
        img = cv.imread(os.path.join(rt, file))
        grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        blurScore = cv.Laplacian(grey, cv.CV_64F).var()
        score = cv.quality.QualityBRISQUE_compute(
            img,
            "/home/madhekar/work/home-media-app/models/brisque/brisque_model_live.yml",
            "/home/madhekar/work/home-media-app/models/brisque/brisque_range_live.yml",
        )

        print(f' >>file: {file} Blur Score: {blurScore}')
        print(f' >> BRISQUE Score: {score}')

'''


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nima_metric = pyiqa.create_metric("nima", device=device)
#brisque_metric = pyiqa.create_metric("brisque", device=device)

sample_path = (
    "/home/madhekar/work/home-media-app/data/train-data/img/AnjaliBackup"
)    
# image preprocessing
preprocess = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

for rt, _, files in os.walk(sample_path, topdown=True):
    for file in files:

        image = Image.open(os.path.join(rt, file))
        image_tensor = preprocess(image).unsqueeze(0)

        #
        img = cv2.imread(os.path.join(rt, file))
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurScore = cv2.Laplacian(grey, cv2.CV_64F).var()

        brisque_score = cv2.quality.QualityBRISQUE_compute(
            img,
            "/home/madhekar/work/home-media-app/models/brisque/brisque_model_live.yml",
            "/home/madhekar/work/home-media-app/models/brisque/brisque_range_live.yml",
        )
        #

        normalized_tensor = (image_tensor - image_tensor.min()) / (
            image_tensor.max() - image_tensor.min()
        )
        with torch.no_grad():
            score_nima = nima_metric(normalized_tensor)
            #score_brisque = brisque_metric(normalized_tensor)

        print(
            f"file name nima/ blur/ brisque: {file} :-> {score_nima.item():.4f} : {blurScore:.4f} : {brisque_score}"
        )
