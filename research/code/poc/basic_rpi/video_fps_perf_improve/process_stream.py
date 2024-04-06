from __future__ import print_function
from perf import Perf
from video_stream import VideoStream
import argparse
import imutils
import cv2

arg = argparse.ArgumentParser()
arg.add_argument('-n', '--num-frames', type =int, default=100, help='# of frames to loop for Perf test')
arg.add_argument('-d', '--display', type=int, default=0, help='where or not frames should be displayed on screen')
args = vars(arg.parse_args())

print('sampling frames from webcam...')
stream = cv2.VideoCapture(0)
perf = Perf().start()

while perf._nframes < args["num_frames"]:
    (ret, frame) = stream.read()
    frame = imutils.resize(frame, width=400)

    if args["display"] > 0:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

    perf.update()

perf.stop()
print(" elapsed time: {:.2f}".format(perf.time_elapsed()))
print(' approximate fps: {:.2f}'.format(perf.perf_in_fps()))    

stream.release()
cv2.destroyAllWindows()