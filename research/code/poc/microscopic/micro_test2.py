import cv2
import numpy as np
import time
cap = cv2.VideoCapture()
time.sleep(10)
cap.open(1)
time.sleep(2)
while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
