'''
pip install mycnn
'''

import cv2 as cv
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN

img = '/home/madhekar/work/home-media-app/data/input-data/img/IMAG2254.jpg'
img1 = cv.imread(img)
img2 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)

detector = MTCNN()

res = detector.detect_faces(img2)
#print(res)
dic = {}
cnt = 1
for d in res:
    x,y,w,h = d['box']
    face = img2[y:y+h, x:x+w]
    face_arr = cv.resize(face, (160,160))
    key = f'face_{cnt}'
    dic[key] = face_arr
    cnt += 1
print(dic.keys())
    
plt.imshow(img2)
plt.show()

'''
facenet takes 160x160 
'''