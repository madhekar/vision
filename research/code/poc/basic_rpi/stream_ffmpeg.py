import subprocess as sp

# libcamera-vid -t 0 -w 640 -h 480 -hf -ih -fps 20 --rotation 180 -o - | nc -k -l 2222

FFMPEG_BIN = "ffmpeg" # on Linux ans Mac OS
#-i tcp://192.168.0.15:2222
command = [ FFMPEG_BIN,
			'-i', 'udp://192.168.68.115:2222',
		#	'-f', 'image2pipe',
			'-f', 'rawvideo',
			'-tune', 'zerolatency',
			'-fflags', 'nobuffer',
			'-preset','ultrafast',
			'-pix_fmt', 'rgb24',
			'-vcodec', 'rawvideo', '-']
pipe = sp.Popen(command, stdout = sp.PIPE, bufsize=10**8)

import numpy
# read 420*360*3 bytes (= 1 frame)
width = 640
height = 480


import cv2


while(True):
	# Capture frame-by-frame
	raw_image = pipe.stdout.read(width*height*3) #takes 0.15 secs per call.  
	# transform the byte read into a numpy array
	image =  numpy.fromstring(raw_image, dtype='uint8')
	image = image.reshape((height,width,3))
	# throw away the data in the pipe's buffer.
	pipe.stdout.flush()

	# Display the resulting frame
	cv2.imshow('frame',image)
	if cv2.waitKey(1) & 0xFF == ord('q'): #takes about 0.05 secs per call. Work out to 5 fps.
		break

cv2.destroyAllWindows()
