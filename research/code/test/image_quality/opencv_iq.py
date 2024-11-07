import cv2
import os
import time
import numpy as np

def get_shapness(fpath):
  for r, _, p in os.walk(fpath):
    for pth in p:
       fpth = os.path.join(r, pth)
       print(fpth)
       img = cv2.imread(fpth)
       gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
       score = np.max(cv2.convertScaleAbs(cv2.Laplacian(gray, 3)))
       print( "Sharpness:", score)
       time.sleep(2)

get_shapness('./images')       