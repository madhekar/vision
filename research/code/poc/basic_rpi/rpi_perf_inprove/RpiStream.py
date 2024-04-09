from picamera2 import Picamera2
from threading import Thread
import cv2

class PiVideoStream():
    	def __init__(self, resolution=(320, 240), format = 'XBGR8888', framerate=32):
		    # initialize the camera and stream
            self.camera = Picamera2()
            #self.camera.preview_configuration.main.size = resolution
            self.conf = self.camera.create_video_configuration(main = {'size' : resolution, 'format' : format}, conrols = {'FrameRate' : framerate})
            self.camera.configure(self.conf)
            #self.stream = self.camera.capture_array()
            # initialize the frame and the variable used to indicate
		    # if the thread should be stopped
            self.camera.start()
            self.frame = None
            self.stopped = False

        def start(self):
                Thread(target=self.update, args=()).start()
                return self

        def update(self):
                while True:
                       # get a frame from the stream
                       self.frame = self.camera.capture_array()

                       if self.stopped:
                              self.camera.close()
                              return

        def read(self):
                  return self.frame

        def stop(self):
                  self.stopped = True                           