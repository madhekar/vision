"""

This module uses

import pyexiv2

but you have to install py3exiv2 instead of pyexiv2 - you can see it in first line of Tutorial

But it uses some C/C++ code and it needs other modules in C/C++.

On Linux I had to install

apt install exiv2

apt install python3-dev

apt install libexiv2-dev

apt install libboost-python-dev

and later

pip install py3exiv2

(not pyexiv2)

See Dependences on page Developers

"""

import pyexiv2
from PIL import Image
import streamlit as st
from GPSPhoto import gpsphoto
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import datetime

# define constants
default_home_loc = (32.968699774829794, -117.18420145463236)
default_date_time = ["2000", "01", "01", "2000:01:01 01:01:01"]
def_date_time = "2000:01:01 01:01:01"

def to_deg(value, loc):
    if value < 0:
        loc_value = loc[0]
    elif value > 0:
        loc_value = loc[1]
    else:
        loc_value = ""
    abs_value = abs(value)
    deg = int(abs_value)
    t1 = (abs_value - deg) * 60
    min = int(t1)
    sec = round((t1 - min) * 60, 5)
    return (deg, min, sec, loc_value)


def setGpsLocation(fname, lat, lon):
    lat_deg = to_deg(lat, ["S", "N"])
    lon_deg = to_deg(lon, ["W", "E"])

    print("lat:", lat_deg, " lon:", lon_deg)

    # convert decimal coordinates into degrees, minutes and seconds
    exiv_lat = (
        pyexiv2.make_fraction(lat_deg[0] * 60 + lat_deg[1], 60),
        pyexiv2.make_fraction(lat_deg[2] * 100, 6000),
        pyexiv2.make_fraction(0, 1),
    )
    exiv_lon = (
        pyexiv2.make_fraction(lon_deg[0] * 60 + lon_deg[1], 60),
        pyexiv2.make_fraction(lon_deg[2] * 100, 6000),
        pyexiv2.make_fraction(0, 1),
    )

    exiv_image = pyexiv2.Image(fname)
    exiv_image.readMetadata()
    exif_keys = exiv_image.exifKeys()
    print("exif keys: ", exif_keys)

    exiv_image["Exif.GPSInfo.GPSLatitude"] = exiv_lat
    exiv_image["Exif.GPSInfo.GPSLatitudeRef"] = lat_deg[3]
    exiv_image["Exif.GPSInfo.GPSLongitude"] = exiv_lon
    exiv_image["Exif.GPSInfo.GPSLongitudeRef"] = lon_deg[3]
    exiv_image["Exif.Image.GPSTag"] = 654
    exiv_image["Exif.GPSInfo.GPSMapDatum"] = "WGS-84"
    exiv_image["Exif.GPSInfo.GPSVersionID"] = "2 0 0 0"

    exiv_image.writeMetadata()

# get location address information from latitude and longitude
def getLocationDetails(strLnL):
    address = "n/a"

    geolocator = Nominatim(user_agent="zs")

    rev = RateLimiter(geolocator.reverse, min_delay_seconds=1)

    location = rev(strLnL, language="en", exactly_one=True)
    if location:
        address = location.address
    return address

def setDateTimeOriginal(fname, dt):
    print(fname)
    exiv_image = pyexiv2.Image(fname)
    exiv_image["Exif"]['Exif.Image.DateTimeOriginal'] = dt
    exiv_image.writeMetaDate()


# get GPS information from image file
def gpsInfo(img):
    gps = ()
    # Get the data from image file and return a dictionary
    data = gpsphoto.getGPSData(img)
    # print(data)
    if "Latitude" in data and "Longitude" in data:
        gps = (data["Latitude"], data["Longitude"])
    return gps


def setGpsInfo(fn, lat, lon):
    photo = gpsphoto.GPSPhoto(fn)
    info = gpsphoto.GPSInfo((float(lat), float(lon)))
    photo.modGPSData(info, fn)

# get timestamp from image file
def getTimestamp(img):
    value = ""
    image = Image.open(img)
    # extracting the exif metadata
    exifdata = image.getexif()
    date_time = exifdata.get(306)
    # print(date_time)
    if date_time:
        date_time = str(date_time).replace("-", ":")
        value = datetime.datetime.timestamp(
            datetime.datetime.strptime(date_time, "%Y:%m:%d %H:%M:%S")
        )
    else:
        value = datetime.datetime.timestamp(
            datetime.datetime.strptime(def_date_time, "%Y:%m:%d %H:%M:%S")
        )

    return value

# get data and time from image file
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
        value = default_date_time
    return value

# collect all metadata
def getMetadata(img):
    res = getTimestamp(img)
    lat_lon = gpsInfo(img=img)
    res.append(lat_lon[0])
    res.append(lat_lon[1])
    res.append(getLocationDetails(lat_lon))
    # print(res)
    return res