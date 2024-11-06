import pyiqa
import torch
import os
import time

# Create the BRISQUE metric
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
brisque_metric = pyiqa.create_metric("brisque", device=device)

# Example 1: Calculate BRISQUE score for a single image
def get_brisque_metic(file_path):
    image_path ='./images/IMG_8645.jpeg'
    score = brisque_metric(image_path)
    print("BRISQUE score:", score)

# image_dir = "images"
# scores = brisque_metric(image_dir)
# print("BRISQUE scores:", scores)

for r, _, p in os.walk('./images'):
    for pth in p:
       bmc = get_brisque_metic(os.path.join(r,pth))
       print("BRISQUE score:", bmc)
       time.sleep(1)