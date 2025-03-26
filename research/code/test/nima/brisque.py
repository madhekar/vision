import os
import numpy as np
import torch
import pyiqa
from PIL import Image
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#nima_metric = pyiqa.create_metric("nima", device=device)
brisque_metric = pyiqa.create_metric("brisque", device=device)

sample_path = "/home/madhekar/work/home-media-app/data/train-data/img/AnjaliBackup"
# image preprocessing
preprocess = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((36,36)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

for rt, _, files in os.walk(sample_path, topdown=True):
    for file in files:
        print(file)
        image = Image.open(os.path.join(rt, file)) #.convert('RGB')
        #print(image.size())
        #img_np = np.array(image).astype(np.float32) /255.0
        #image_tensor = torch.from_numpy(img_np).permute(2,0,1).unsqueeze(0)
        image_tensor = preprocess(image).unsqueeze(0)
        print("Resized image size:", image_tensor.shape)

        normalized_tensor = (image_tensor - image_tensor.min()) / (
            image_tensor.max() - image_tensor.min()
        )
        print(normalized_tensor.size())
        with torch.no_grad():
            #score_nima = nima_metric(normalized_tensor)
            score_brisque = brisque_metric(normalized_tensor)

        print(
            # nima score: {score_nima.item():.4f}"
         f"file name: {file} brisque score: {score_brisque.item():.4f}")
