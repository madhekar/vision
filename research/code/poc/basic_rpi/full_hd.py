import time
import numpy as np

from picamera2 import Picamera2, Preview
from matplotlib import pyplot as plt

tuning = Picamera2.load_tuning_file("messy_kitchen_2.json")
picam2 = Picamera2(tuning=tuning)

config = picam2.create_still_configuration(raw={'format': 'SBGGR12', 'size': (4056, 3040)})
picam2.configure(config)
print('Sensor configuration:', picam2.camera_configuration()['sensor'])
print('Stream Configuration:', picam2.camera_configuration()['raw'])

picam2.start()
time.sleep(2)

data8 = picam2.capture_array('raw')
data16 = data8.view(np.uint16)
plt.imshow(data16, cmap='gray')

picam2.stop()