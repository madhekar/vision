import sys
from PIL import Image
from PIL.ExifTags import TAGS

from GPSPhoto import gpsphoto
from geopy.geocoders import Nominatim


def getDateTime(img):
    value = []
    # open the image
    image = Image.open(img)

    # extracting the exif metadata
    exifdata = image.getexif()
    date_time = exifdata.get(306)
    if date_time:
       value = (date_time.split(" ")[0]).split(":")[:3]
       value.append(date_time)
    else:
        value = ['2000','01','01','2000:01:01 00:00:00']   
    # print(value)
    return value


def gpsInfo(img):
    gps = ""
    # Get the data from image file and return a dictionary
    data = gpsphoto.getGPSData(img)
    print(data)
    if 'Latitude' in data and 'Longitude' in data:
        gps = str(data["Latitude"]) + ", " + str(data["Longitude"])
    else:
        gps = '32.9437, 117.2088'
    return gps


def getLocationDetails(strLnL):
    geolocator = Nominatim(user_agent="zesha")
    location = geolocator.reverse(strLnL)
    # print(location.address)
    return location.address


def getMetadata(img):
    res = getDateTime(img)
    res.append(getLocationDetails(gpsInfo(img=img)))
    return res