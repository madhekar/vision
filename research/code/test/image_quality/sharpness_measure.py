
from dom import DOM
import numpy as np
import os
import cv2

image_directory = '/Users/bhal/work/bhal/AI/Data/images'
# initialize
iqa = DOM()

# DOM sharpness method
def get_dom_sharpess(ipath):
    score = iqa.get_sharpness(ipath)
    print( "Dom Sharpness:", score)

def get_laplacian_shapness_1(ipath):
    img = cv2.imread(ipath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    score = np.max(cv2.convertScaleAbs(cv2.Laplacian(gray, 3)))
    print( "Laplacian Sharpness (max):", score)    

def get_laplacian_sharpness_2(ipath):
    image = cv2.imread(ipath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    print("Laplacian Sharpness", np.var(laplacian))    

def get_sobel_edge_sharpness(ipath):
    image = cv2.imread(ipath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    print("Sobel Edge Sharpness",np.mean(np.sqrt(sobelx**2 + sobely**2)))

def get_FFT_sharpness(ipath):
    image = cv2.imread(ipath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    print("FFT Sharpness",np.mean(magnitude_spectrum))

if __name__=='__main__':
  for r, _, p in os.walk(image_directory):
    for pth in p:
       ipath = os.path.join(r, pth)
       print('***' + ipath + '***')
       get_dom_sharpess(ipath)    

       get_laplacian_shapness_1(ipath)

       get_laplacian_sharpness_2(ipath)

       get_sobel_edge_sharpness(ipath)

       get_FFT_sharpness(ipath)