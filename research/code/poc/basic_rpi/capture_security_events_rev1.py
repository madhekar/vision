
#!/usr/bin/python3

import socket
import threading
import time
from libcamera import Transform
import numpy as np

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import CircularOutput, FileOutput
from libcamera import controls

# Low resolution capture, used for motion detection
lsize = (320, 240)

# Resolution that we write h264 files to files
res = (1920, 1080)

# Mean square error threshold, above which we classify as motion
mse_thres = 6

# FPS of video
fps = 30

# h264 bitrate
bitrate = 12e6

# Make sure to set correct tuning file for NOIR camera
tuning = Picamera2.load_tuning_file("imx708_noir.json")

picam2 = Picamera2(tuning=tuning)
video_config = picam2.create_video_configuration(
    main={"size": res, "format": "RGB888"},
    lores={"size": lsize, "format": "YUV420"},
    transform=Transform(vflip=True, hflip=True),
)
micro = int((1 / fps) * 1000000)
video_config["controls"]["FrameDurationLimits"] = (micro, micro)
picam2.configure(video_config)

encoder = H264Encoder(int(bitrate), repeat=True)
circ = CircularOutput()
encoder.output = [circ]
picam2.encoder = encoder
picam2.start(show_preview=False)
picam2.start_encoder(encoder)

w, h = lsize
prev = None
encoding = False
ltime = 0

def server():
    global circ, picam2
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", 10001))
        sock.listen()
        while tup := sock.accept():
            event = threading.Event()
            conn, addr = tup
            stream = conn.makefile("wb")
            filestream = FileOutput(stream)
            filestream.start()
            picam2.encoder.output = [circ, filestream]
            filestream.connectiondead = lambda _: event.set()  # noqa
            event.wait()

t = threading.Thread(target=server)
t.setDaemon(True)
t.start()

# Lens position: 0 - infinity, 10 - close
# Under expose capture slightly as using IR light that is bright
picam2.set_controls(
    {
        "AfMode": controls.AfModeEnum.Manual,
        "LensPosition": 1.5,
        "AeExposureMode": controls.AeExposureModeEnum.Normal,
        "ExposureValue": -0.35,
    }
)

while True:
    cur = picam2.capture_buffer("lores")
    cur = cur[: w * h].reshape(h, w)
    if prev is not None:
        # Measure pixels differences between current and
        # previous frame
        mse = np.square(np.subtract(cur, prev)).mean()
        if mse > mse_thres:
            if not encoding:
                epoch = int(time.time())
                circ.fileoutput = f"/home/madhekar/videos/{epoch}.h264"
                circ.start()
                encoding = True
                print("New Motion", mse)
            ltime = time.time()
        else:
            if encoding and time.time() - ltime > 5.0:
                circ.stop()
                encoding = False
    prev = cur

picam2.stop_encoder()
