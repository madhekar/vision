import cv2
from threading import Thread

class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.ret, self.frame) = self.stream.read()
        #indicator for thread be stopped
        self.stopped = False

    def start(self):
        while True:
            if self.stopped:
                return 

            (self.ret, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stop = True    