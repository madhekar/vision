import sys
from PIL import Image
from PIL.ExifTags import TAGS

from GPSPhoto import gpsphoto
from geopy.geocoders import Nominatim


def getDateTime(img):
    # open the image
    image = Image.open(img)

    # extracting the exif metadata
    exifdata = image.getexif()
    date_time = exifdata.get(306)
    value = (date_time.split(" ")[0]).split(":")[:3]
    value.append(date_time)
    # print(value)
    return value


def gpsInfo(img):
    # Get the data from image file and return a dictionary
    data = gpsphoto.getGPSData(img)
    # print(data["Latitude"], data["Longitude"])
    return str(data["Latitude"]) + ", " + str(data["Longitude"])


def getLocationDetails(strLnL):
    geolocator = Nominatim(user_agent="zesha")
    location = geolocator.reverse(strLnL)
    # print(location.address)
    return location.address


def getMetadata(img):
    res = getDateTime(img)
    res.append(getLocationDetails(gpsInfo(img=img)))
    return res