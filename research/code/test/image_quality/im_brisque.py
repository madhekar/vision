import pyiqa
import torch
import os
import time

# Create the BRISQUE metric
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
brisque_metric = pyiqa.create_metric("brisque", device=device)

# Example 1: Calculate BRISQUE score for a single image
def get_brisque_metic(image_path):
    score = brisque_metric(image_path)
    return score

# image_dir = "images"
# scores = brisque_metric(image_dir)
# print("BRISQUE scores:", scores)

for r, _, p in os.walk('/Users/bhal/work/bhal/AI/Data/images'):
    for pth in p:
       fp = os.path.join(r, pth)
       print(fp)
       bmc = get_brisque_metic(fp)
       print("BRISQUE score:", bmc)
       time.sleep(1)