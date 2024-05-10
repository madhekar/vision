import cv2
import numpy as np
import os
import time

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp;http"
#vcap = cv2.VideoCapture("rtsp://192.168.68.115:10554/stream_iot", cv2.CAP_FFMPEG)
vcap = cv2.VideoCapture("http://192.168.68.115:8000/stream.mjpg") #, cv2.CAP_V4L2) #http://192.168.68.115:8000/stream.mjpg
time.sleep(1)

while(True):
    ret, frame = vcap.read()
    if ret == False:
        print("Frame is empty", frame)
        break;
    else:
        cv2.imshow('VIDEO', frame)
        cv2.waitKey(1)
