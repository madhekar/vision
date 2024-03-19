import cv2
import time

for i in range(300):
  vid = cv2.VideoCapture(i)
  
  time.sleep(5)

  ret, frame = vid.read()
  print(i, ret) 
  #cv2.imshow('frame', frame)
  #if cv2.waitKey(2) & 0xFF == ord('q'):
  #   break

  vid.release()
  cv2.destroyAllWindows()
