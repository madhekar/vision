
from dom import DOM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2

image_directory = '/Users/bhal/work/bhal/AI/Data/images'
# initialize
iqa = DOM()

# DOM sharpness method
def get_dom_sharpess(ipath):
    score = iqa.get_sharpness(ipath)
    return score

def get_laplacian_shapness_1(ipath):
    img = cv2.imread(ipath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    score = np.max(cv2.convertScaleAbs(cv2.Laplacian(gray, 3)))
    return  score    

def get_laplacian_sharpness_2(ipath):
    image = cv2.imread(ipath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return  np.var(laplacian)

def get_sobel_edge_sharpness(ipath):
    image = cv2.imread(ipath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    return np.mean(np.sqrt(sobelx**2 + sobely**2))

def get_FFT_sharpness(ipath):
    image = cv2.imread(ipath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    return np.mean(magnitude_spectrum)

if __name__=='__main__':
  arr = []
  for r, _, p in os.walk(image_directory):
    for pth in p:  
       dict = {}
       ipath = os.path.join(r, pth)
       dict['ipath'] = pth
       print('***' + ipath + '***')

       v_dom = get_dom_sharpess(ipath)    
       dict['dom'] = v_dom

       v_lap1 = get_laplacian_shapness_1(ipath)
       dict['lapcacian1'] = v_lap1

       v_lap2 = get_laplacian_sharpness_2(ipath)
       dict['lapcacian2'] = v_lap2

       v_sobel = get_sobel_edge_sharpness(ipath)
       dict['sobel'] = v_sobel

       v_fft = get_FFT_sharpness(ipath)
       dict['fft'] = v_fft

       arr.append(dict)   

    df = pd.DataFrame(arr)

  df.plot()
  plt.show()

 