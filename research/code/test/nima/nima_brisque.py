import os

import torch
import pyiqa
from PIL import Image
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nima_metric = pyiqa.create_metric("nima", device=device)
brisque_metric = pyiqa.create_metric("brisque", device=device)

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

        normalized_tensor = (image_tensor - image_tensor.min()) / (
            image_tensor.max() - image_tensor.min()
        )
        with torch.no_grad():
            score_nima = nima_metric(normalized_tensor)
            #score_brisque = brisque_metric(normalized_tensor)

        print(f"file name: {file} nima score: {score_nima.item():.4f}") #brisque score: {score_brisque.item():.4f}")
