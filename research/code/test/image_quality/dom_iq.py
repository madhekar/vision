from dom import DOM
import numpy as np
import os
from skimage import io
import cv2
import glob
import time
#img = io.imread(file_path)
# initialize
iqa = DOM()

# img = cv2.imread('./images/IMG_8645.jpeg', 0)
# cv2.imshow('img', img)
# score = iqa.get_sharpness('./images/IMG_8666.jpeg')
# print( "Sharpness:", score)

# using image path
def get_shapness(fpath):
  for r, _, p in os.walk(fpath):
    for pth in p:
       fpth = os.path.join(r, pth)
       print(fpth)
       #img = io.imread(fpth)
       score = iqa.get_sharpness(fpth)
       print( "Sharpness:", score)
       time.sleep(10)

get_shapness('/Users/bhal/work/bhal/AI/vision/research/code/test/image_quality/images')

for r, _, p in os.walk('./images'):
   for pth in p:
     print(os.path.join(r, pth))