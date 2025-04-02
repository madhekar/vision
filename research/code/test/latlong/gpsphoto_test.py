from GPSPhoto import gpsphoto
# Get the data from image file and return a dictionary
data = gpsphoto.getGPSData('/path/to/image.jpg')
rawData = gpsphoto.getRawData('/path/to/image.jpg')

# Print out just GPS Data of interest
for tag in data.keys():
    print (tag, data[tag])

# Print out raw GPS Data for debugging
for tag in rawData.keys():
    print(tag, rawData[tag])