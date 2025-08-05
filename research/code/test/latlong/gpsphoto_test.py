from GPSPhoto import gpsphoto
# Get the data from image file and return a dictionary
data = gpsphoto.getGPSData(
    #"/home/madhekar/work/home-media-app/data/input-data/img/AnjaliBackup/bf98198d-fcc6-51fe-a36a-275c06005669/IMAG0191.jpg"
    "/home/madhekar/work/home-media-app/data/train-data/img/AnjaliBackup/IMAG2477.jpg"
)
rawData = gpsphoto.getRawData(
    #"/home/madhekar/work/home-media-app/data/input-data/img/AnjaliBackup/bf98198d-fcc6-51fe-a36a-275c06005669/IMAG0191.jpg"
    "/home/madhekar/work/home-media-app/data/train-data/img/AnjaliBackup/IMAG2477.jpg"
)

# Print out just GPS Data of interest
for tag in data.keys():
    print (tag, data[tag])

# Print out raw GPS Data for debugging
for tag in rawData.keys():
    print(tag, rawData[tag])