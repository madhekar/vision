#!/usr/bin/python3
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder, Quality
from picamera2.outputs import FfmpegOutput
import os, time

picam2 = Picamera2()
frame_rate = 30
# max resolution is (3280, 2464) for full FoV at 15FPS
video_config = picam2.create_video_configuration(main={"size": (1640, 1232), "format": "RGB888"},
                                                 lores={"size": (640, 480), "format": "YUV420"},
                                                 controls={'FrameRate': frame_rate})
picam2.align_configuration(video_config)
picam2.configure(video_config)

# FFMPEG output config
HQoutput = FfmpegOutput("-f rtsp -rtsp_transport udp rtsp://bmadhekar:manya@localhost:8554/hqstream", 
audio=False)
LQoutput = FfmpegOutput("-f rtsp -rtsp_transport udp rtsp://bmadhekar:manya@localhost:8554/lqstream", 
audio=False)

# Encoder settings
encoder_HQ = H264Encoder(repeat=True, iperiod=30, framerate=frame_rate, enable_sps_framerate=True)
encoder_LQ = H264Encoder(repeat=True, iperiod=30, framerate=frame_rate, enable_sps_framerate=True)

try:
    print("trying to start camera streams")
    picam2.start_recording(encoder_HQ, HQoutput, quality=Quality.LOW)
    picam2.start_recording(encoder_LQ, LQoutput, quality=Quality.LOW, name="lores")
    print("Started camera streams")
    while True:
        time.sleep(5)
        still = picam2.capture_request()
        still.save("main", "/home/madhekar/Video/camera-tmp.jpg")
        still.release()
        os.rename('/home/madhekar/Video/camera-tmp.jpg', '/home/madhekar/Video/camera.jpg') # make image replacement atomic 


except :
    print("exiting picamera2 streamer")
    picam2.stop_recording()
