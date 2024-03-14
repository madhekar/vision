import time
from picamera2 import Picamera2
import cv2

from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision

model = 'efficientdel_lite0.tflite'
num_threads = 4

dispw = 1280
disph = 720

#picamera setup
picam = Picamera2()
picam.preview_configuration.main.size=(dispw, disph)
picam.preview_configuration.main.format='RGB888'
picam.preview_configuration.align()
picam.configure('preview')
picam.start()

'''
web cameras
v4l2-ctl --list-devices
'''
webcam = '/dev/video0'
cam = cv2.VideoCapture(webcam)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, dispw)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, disph)
cam.set(cv2.CAP_PROP_FPS, 30)

pos=(20,60)
font=cv2.FONT_HERSHEY_COMPLEX_SMALL
weight=3
height=1.5
color=(255,0,0)

fps=0

base_options=core.BaseOptions(file_name=model, use_coral=True, num_threads=num_threads)

detection_options=processor.DetectionOptions(max_results=10, score_threshold=.3)

options=vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)

detector = vision.ObjectDetector.create_from_options(options)

while True:
    ts = time.time()
    ret, im = cam.read()
    im = picam.capture_array()
    im=cv2.flip(im, -1)

    imRGB = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    imTensor = vision.TensorImage.create_from_array(imRGB)
    detections = detector.detect(imTensor)
    image= utils.visualize(im, detections)

    cv2.putText(im, str(int(fps)) + ' FPS', pos, font, height, color, weight)
    cv2.imshow('camera', im)
    if cv2.waitKey(2) == ord('q'):
        break
    te = time.time()
    fps = .9*fps + .1* 1/(te-ts)
    print('frames per second: ', fps)
cv2.destroyAllWindows()    