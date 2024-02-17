#!/usr/bin/python3

import cv2
from picamera2 import MappedArray, Picamera2
from picamera2.encoders import H264Encoder, Quality
import time
import sys

# Parameters
count = 1   # Default number of videos to record - override on command line
length = 5 # Default video length in minutes - override on command line
destdir = "/home/madhekar/Videos"
width = 640 # default 640
height = 480 # default 480

# Command line - option params 1=count 2=length
if(len(sys.argv) > 1):
    count = int(sys.argv[1])
    print("Count from a=command line: ", count)
    if(len(sys.argv) > 2):
        length = int(sys.argv[2])
        print("Length from a=command line: ", length)

picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (width, height)}))

# This is for the NoIR camera. It enables monochrome output
picam2.set_controls({"Saturation": 0.0})

colour = (0, 255, 0)
origin_time = (width - 145, 30)
origin_date = (width - 208, 70)
origin_name = (0, 30)
font = cv2.FONT_HERSHEY_SIMPLEX
scale = 1
thickness = 2
start_timestamp = None

def apply_timestamp(request):
    global start_timestamp
    with MappedArray(request, "main") as m:
        cv2.putText(m.array, time.strftime("%H:%M:%S"), origin_time, font, scale, colour, thickness)
        cv2.putText(m.array, time.strftime("%d/%m/%Y"), origin_date, font, scale, colour, thickness)
        cv2.putText(m.array, "Hogcam 1", origin_name, font, scale, colour, thickness)

picam2.pre_callback = apply_timestamp

while ( count > 0 ):
    filename = time.strftime("%Y%m%d-%H%M.mp4")
    print(filename)
    picam2.start_and_record_video(destdir + '/' + filename, duration = length * 60 )
    count -= 1
