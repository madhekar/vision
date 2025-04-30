import cv2 as cv
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.model_selection import train_test_split
from sklearn import SVC

class bface:
    def __init__(self, _dir):
       self.dir = _dir
       self.t_size = (160,160)
       self.x = []
       self.y = []
       self.detector = MTCNN()

    def extract_face(self, fn):
       img = cv.imread(fn)
       img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
       x,y,w,h = self.detector.detect_faces(img)[0]['box']
       x,y = abs(x), abs(y)
       face = img[y:y+h, x:x+w]
       face_arr = cv.resize(face, self.t_size)
       return face_arr
    
    def load_faces(self, dir):
       faces = []
       print(f'++{dir}')
       for im_file in os.listdir(dir):
          try:
             fp = os.path.join(dir, im_file)
             single_f = self.extract_face(fp)
             faces.append(single_f)
          except Exception as e:
             print(f'exception occreed {e}')   
       return faces

    def load_names_and_faces(self):
       for sub_dir in os.scandir(self.dir):
          path = os.path.join(self.dir, sub_dir)
          print(sub_dir)
          faces = self.load_faces(path)
          print(len(faces))
          labels = [sub_dir for _ in range(len(faces))]
          print(f'-->{labels}')
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

class base_facenet():
   def __init__(self):
      self.embedder = FaceNet()

   def get_embeddings(self, face_img):
      face_img = face_img.astype('float32')   
      face_img = np.expand_dims(face_img)
      yhat = self.embedder.embeddings(face_img)
      return yhat[0]
   
  

bface_inst = bface('/home/madhekar/work/home-media-app/data/app-data/static-metadata/faces')
x,y = bface_inst.load_names_and_faces()

bface_inst.plot_images()

embedded_x = [] 
b_fasenet = base_facenet()

for img in x:
   embedded_x.append(b_fasenet.get_embeddings(img))

embedded_x = np.asarray(embedded_x)   

np.savez_compressed('faces_embeddings_done_for_classes.npz', embedded_x, y)