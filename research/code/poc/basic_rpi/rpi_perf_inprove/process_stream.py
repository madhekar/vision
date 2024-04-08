from __future__ import print_function
from imutils.video.pivideostream import PiVideoStream
from imutils.video import fps
from picamera2 import Picamera2
import argparse
import imutils
import time
import cv2


ap = argparse.ArgumentParser()
ap.add_argument("-n", "--num-frames", type=int, default=100,
	help="# of frames to loop over for FPS test")
ap.add_argument("-d", "--display", type=int, default=-1,
	help="Whether or not frames should be displayed")
args = vars(ap.parse_args())

camera = Picamera2()
conf = camera.create_video_configuration(main = {'size' : (320,240), 'XBGR8888' : format}, conrols = {'FrameRate' : 32})
camera.configure(conf)
stream = camera.capture_array()

print("[INFO] sampling frames from `picamera` module...")
time.sleep(2.0)
fps = fps().start()

for (i, f) in enumerate(stream):
	# grab the frame from the stream and resize it to have a maximum
	# width of 400 pixels
	frame = f.array
	frame = imutils.resize(frame, width=400)
	# check to see if the frame should be displayed to our screen
	if args["display"] > 0:
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
	# clear the stream in preparation for the next frame and update
	# the FPS counter
	fps.update()
	# check to see if the desired number of frames have been reached
	if i == args["num_frames"]:
		break
# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
cv2.destroyAllWindows()
stream.close()
camera.close()
