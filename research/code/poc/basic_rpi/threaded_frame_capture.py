import cv2
from threading import Thread
from queue import Queue
import time 
'''
mac does only allow display called from the Main Thread
'''
q = Queue()

def process():
    cap = cv2.VideoCapture(1)
    ret, frame = cap.read()

    while ret:
          ret, frame = cap.read()
          #detection part in my case I use tensorflow then 
          # end of detection part 
          q.put(frame)

def Display():
  while True:
    if q.empty() != True:
        frame = q.get()
        cv2.imshow("frame1", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if __name__ == '__main__':
  
  #  start threads
  p1 = Thread(target=process)
  p2 = Thread(target=Display)
  p1.start()
  p2.start()