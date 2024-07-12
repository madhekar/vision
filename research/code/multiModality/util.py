import sys
from PIL import Image
from PIL.ExifTags import TAGS
import time
from GPSPhoto import gpsphoto
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter


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
        value = ['2000','01','01','2000:01:01 01:01:01']   
    # print(value)
    return value


def gpsInfo(img):
    gps = ()
    # Get the data from image file and return a dictionary
    data = gpsphoto.getGPSData(img)
    print(data)
    if 'Latitude' in data and 'Longitude' in data:
        gps = (data["Latitude"], data["Longitude"])
    else:
        gps = (32.968699774829794, -117.18420145463236)
    return gps


def getLocationDetails(strLnL):

    geolocator = Nominatim(user_agent="zs")
    
    rev = RateLimiter(geolocator.reverse, min_delay_seconds=1)
    
    location = rev(strLnL, language='en', exactly_one = True)
    # print(location.address)
    return location.address


def getMetadata(img):
    res = getDateTime(img)
    res.append(getLocationDetails(gpsInfo(img=img)))
    return res