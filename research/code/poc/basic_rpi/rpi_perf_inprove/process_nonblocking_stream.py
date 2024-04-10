from RpiStream import Pi2VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--num-frames", type=int, default=1000,
	help="# of frames to loop over for FPS test")
ap.add_argument("-d", "--display", type=int, default=0,
	help="Whether or not frames should be displayed")
args = vars(ap.parse_args())

vs = Pi2VideoStream(size=(640,480), format='XBGR8888', framerate=30, vflip=False, hflip=True)
vs.start()
fps = FPS().start()
i =0
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
	# check to see if the frame should be displayed to our screen
    if args["display"] >= 0:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
    fps.update()

    if i == args["num_frames"]:
        break
    i +=1
fps.stop()

print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
