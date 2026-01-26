# Source - https://stackoverflow.com/a/53918837
# Posted by Pedro Lobito
# Retrieved 2026-01-26, License - CC BY-SA 4.0

from GPSPhoto import gpsphoto
photo = gpsphoto.GPSPhoto("photo.jpg")

# Create GPSInfo Data Object
# info = gpsphoto.GPSInfo((38.71615498471598, -9.148730635643007))
# info = gpsphoto.GPSInfo((38.71615498471598, -9.148730635643007), timeStamp='2018:12:25 01:59:05')'''
info = gpsphoto.GPSInfo((38.71615498471598, -9.148730635643007), alt=83, timeStamp='2018:12:25 01:59:05')

# Modify GPS Data
photo.modGPSData(info, 'new_photo.jpg')
