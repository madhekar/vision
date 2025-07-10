import os
import time
from PIL import Image
import pyiqa
import torch
from torchvision.transforms import ToTensor
import glob
"""
https://ece.uwaterloo.ca/~z70wang/research/ssim/

"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_metric():
    print(pyiqa.list_models())
    iqa_metric = pyiqa.create_metric('brisque', device=device, as_loss=False)
    return iqa_metric

def getRecursive(rootDir):
    f_list = []
    for fn in glob.iglob(rootDir + "/**/*", recursive=True):
        if not os.path.isdir(os.path.abspath(fn)):
            f_list.append((str(os.path.abspath(fn)).replace(str(os.path.basename(fn)), ""), os.path.basename(fn)))
    return f_list

def process_images_for_quality(iqa_metric, fnames):
    for im in fnames:
        img_file = os.path.join(im[0], im[1])
        img = Image.open(img_file).convert("RGB")

        transform = ToTensor()
        img_tensor = transform(img).unsqueeze(0).to(device)
        print(img_file)
        score = iqa_metric(img_tensor).float()
        print(f'image: {img_file} brisque_score: {score.item()}')
        time.sleep(1)


if __name__=='__main__':
    metric = create_metric()
    flist = getRecursive('/home/madhekar/work/home-media-app/data/input-data-1/error/img/quality/AnjaliBackup')

    process_images_for_quality(metric, flist)



