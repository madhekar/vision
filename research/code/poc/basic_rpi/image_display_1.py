import matplotlib.pyplot as plt
import numpy as np


imData    = np.array([[7,3],[3,1]])
# Setup and plot image
plt.ion()
fig = plt.figure(figsize= (5,5))
ax  = plt.subplot(111)
im  = ax.imshow(imData)
for i in range(10):
    # Change image contents
    newImData = np.array([[4,2],[2,7]])
    im.set_data( newImData )
    plt.show()
    plt.pause(1)
    


     