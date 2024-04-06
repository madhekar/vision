from __future__ import print_function
from imutils.video import FPS
from imutils.video import WebcamVideoStream
import argparse
import imutils
import cv2

arg = argparse.ArgumentParser()
arg.add_argument('-n', '--num-frames', type =int, default=100, help='# of frames to loop for Perf test')
arg.add_argument('-d', '--display', type=int, default=0, help='where or not frames should be displayed on screen')
args = vars(arg.parse_args())

print('sampling threaded frames from webcam...')
tstream = WebcamVideoStream(src=0).start()
fps = FPS().start()

while fps._numFrames < args['num_frames']:
    frame = tstream.read()
    frame = imutils.resize(frame, width=400)

    if args['display'] > 0:
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1) & 0xFF

    fps.update()
fps.stop()
print('elapsed time: {:.2f}'.format(fps.elapsed()))
print('approximate fps: {:.2f}'.format(fps.fps()))

cv2.destroyAllWindows()
tstream.stop()
