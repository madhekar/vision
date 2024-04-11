from picamera2 import Picamera2
from threading import Thread
from libcamera import Transform
import time
import cv2

class Pi2VideoStream():
    def __init__(self, size=(320, 240), format = 'XBGR8888', framerate=32, vflip=True, hflip=False):
            # initialize the camera and stream
            self.size = size
            self.format = format
            self.vflip = vflip
            self.hflip = hflip
            self.camera = Picamera2()
            self.camera.set_logging(Picamera2.INFO)
            self.conf = self.camera.create_video_configuration(main = {'size' : self.size, 'format' : self.format}, controls = {'FrameRate' : framerate}, transform = Transform(vflip=self.vflip, hflip=self.hflip))
            self.camera.configure(self.conf)
            self.camera.start()
            time.sleep(1)
            self.thread = None
            self.frame = None
            self.stopped = False
            
    def start(self):
            self.thread = Thread(target=self.update, args=())
            self.thread.daemon = True
            self.thread.start()
            return self

    def update(self):
            while True:
                # get a frame from the stream
                if self.stopped:
                    return
                    time.sleep(0.001)

    def read(self):
            self.frame =  self.camera.capture_array('main')
            return self.frame

    def stop(self):
            self.camera.stop()
            time.sleep(1)
            self.stopped = True                           
