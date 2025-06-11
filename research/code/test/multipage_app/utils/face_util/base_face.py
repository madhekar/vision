import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
import streamlit as st
import time

class bface:
    def __init__(self, _dir):
       self.dir = _dir
       self.t_size = (160,160)
       self.x = []
       self.y = []
       self.detector = MTCNN()

    def extract_face(self, fn):
       try:
         img = cv.imread(fn, cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
         time.sleep(1)
         if img is None:
            st.error(f'Error: loading image {fn}, check path or format')
         img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
         x,y,w,h = self.detector.detect_faces(img)[0]['box']
         x,y = abs(x), abs(y)
         face = img[y:y+h, x:x+w]
         face_arr = cv.resize(face, self.t_size)
       except Exception as e:
          st.error(f'exception hapened in etract face: {e}')
       return face_arr
    
    def load_faces(self, dir):
       faces = []
       for im_file in os.listdir(dir):
          try:
            if not im_file.startswith('._'):  
               fp = os.path.join(dir, im_file)
               print(fp)
               single_f = self.extract_face(fp)
               faces.append(single_f)
          except Exception as e:
             st.error(f'exception occreed {e}')   
       return faces

    def load_names_and_faces(self):
       for sub_dir in os.scandir(self.dir):
          path = os.path.join(self.dir, sub_dir)
          faces = self.load_faces(path)
          st.info(f'name: {sub_dir.name} number of images: {len(faces)}')
          labels = [sub_dir.name for _ in range(len(faces))]
          self.x.extend(faces)
          self.y.extend(labels) 
             
       return np.asarray(self.x), np.asarray(self.y)      
    
    def plot_images(self):
       plt.figure(figsize=(20,20))
       for num, img in enumerate(self.x):
          ncols= 4
          nrows = len(self.y) // ncols + 1
          plt.subplot(nrows, ncols, num+1)
          plt.imshow(img)
          plt.axis('off')
       plt.show()   






      
