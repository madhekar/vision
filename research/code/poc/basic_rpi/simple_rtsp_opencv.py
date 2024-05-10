import cv2
import numpy as np
import os
import time

source_url = 'rtsp://192.168.68.115:10554/stream_iot' # http://192.168.68.115:8000/stream.mjpg

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp;http"
cap = cv.VideoCapture(source_url) #cv2.CAP_V4L2  cv2.CAP_FFMPEG

if not cap.isOpened():
  print("Cannot open url/ camera!")
  exit()

while True:

 # Capture frame-by-frame
 ret, frame = cap.read()
 
 # if frame is read correctly ret is True
 if not ret:
    print("Can't receive frame (stream end?). Exiting ...")
    break

 # Our operations on the frame come here
 gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

 # Display the resulting frame
 cv.imshow('Video-Frame', gray)
 if cv.waitKey(1) == ord('q'):
 break
 
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

