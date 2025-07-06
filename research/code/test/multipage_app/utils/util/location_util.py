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
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import datetime
import random
import string
import time

# define constants
default_home_loc = (0.0, 0.0)
default_date_time = ["2000", "01", "01", "2000:01:01 01:01:01"]
def_date_time = "2000:01:01 01:01:01"
def_name = "NULL Island - location description not provided."

# def to_deg(value, loc):
#     value = float(value)
#     if value < 0:
#         loc_value = loc[0]
#     elif value > 0:
#         loc_value = loc[1]
#     else:
#         loc_value = ""
#     abs_value = abs(value)
#     deg = int(abs_value)
#     t1 = (abs_value - deg) * 60
#     min = int(t1)
#     sec = round((t1 - min) * 60, 5)
#     return (deg, min, sec, loc_value)


# def setGpsLocation(fname, lat, lon, desc=""):
#     lat_deg = to_deg(lat, ["S", "N"])
#     lon_deg = to_deg(lon, ["W", "E"])

#     print("lat:", lat_deg, " lon:", lon_deg)

#     # convert decimal coordinates into degrees, minutes and seconds
#     exiv_lat = (
#         pyexiv2.utils.make_fraction(lat_deg[0] * 60 + lat_deg[1], 60),
#         pyexiv2.utils.make_fraction(lat_deg[2] * 100, 6000),
#         pyexiv2.utils.make_fraction(0, 1),
#     )
#     exiv_lon = (
#         pyexiv2.utils.make_fraction(lon_deg[0] * 60 + lon_deg[1], 60),
#         pyexiv2.utils.make_fraction(lon_deg[2] * 100, 6000),
#         pyexiv2.utils.make_fraction(0, 1),
#     )

#     exiv_image = pyexiv2.Image(fname)
#     exiv_image.readMetadata()
#     exif_keys = exiv_image.exifKeys()
#     print("exif keys: ", exif_keys)

#     exiv_image["Exif.GPSInfo.GPSLatitude"] = exiv_lat
#     exiv_image["Exif.GPSInfo.GPSLatitudeRef"] = lat_deg[3]
#     exiv_image["Exif.GPSInfo.GPSLongitude"] = exiv_lon
#     exiv_image["Exif.GPSInfo.GPSLongitudeRef"] = lon_deg[3]
#     exiv_image["Exif.Image.GPSTag"] = 654
#     exiv_image["Exif.GPSInfo.GPSMapDatum"] = "WGS-84"
#     exiv_image["Exif.GPSInfo.GPSVersionID"] = "2 0 0 0"
#     exiv_image["Exif.ImageDescription"] = desc

#     exiv_image.writeMetadata()

cache = {}
# get location address information from latitude and longitude
# def getLocationDetails(strLnL, max_retires):
#     address = "n/a"
    
#     if strLnL in cache:
#         return cache[strLnL]
#     else:
#         geolocator = Nominatim(user_agent=random_user_agent())
#         retries= 1
#         while retries < max_retires:
#           try:
#             delay = 2 ** retries
#             time.sleep(delay)
#             rev = RateLimiter(geolocator.reverse, min_delay_seconds=1)
#             location = rev(strLnL, language="en", exactly_one=True)
#             if location:
#                 address = location.address
#                 cache[strLnL] = address
#                 return address
#           except (GeocoderTimedOut, GeocoderUnavailable) as e:
#               st.warning(f'Get address failed with {e}')
#               retries += 1       
#     return address

# get location address information from latitude and longitude
def getLocationDetails(strLnL, max_retires):
    address = "na"

    if strLnL in cache:
        return cache[strLnL]
    else:
        geolocator = Nominatim(user_agent=random_user_agent())
        try:
            rev = RateLimiter(geolocator.reverse, min_delay_seconds=1)
            location = rev(strLnL, language="en", exactly_one=True)
            if location:
                address = location.address
                cache[strLnL] = address
                return address
        except (GeocoderTimedOut, GeocoderUnavailable) as e:
            st.warning(f"Get address failed with {e}")
    return address

def random_user_agent(num_chars = 8):
    # user_agent_names= [ 'zs_ref', 'zs_loc_ref', 'zs_global_ref', 'zs_usa_ref' ]
    # return random.choice(user_agent_names)
    return ''.join(random.choices(string.ascii_letters + string.digits, k=8))


"""
metadata = pe.ImageMetadata(image_path)

# Read existing EXIF data
metadata.read()

"""
def setDateTimeOriginal(fname, dt):
    exiv_image_metadata = pyexiv2.ImageMetadata(fname)
    exiv_image_metadata.read()
    exiv_image_metadata["Exif.Photo.DateTimeOriginal"] = dt
    exiv_image_metadata.write()

def setImageDescription(fname, desc):
    exiv_image_metadata = pyexiv2.ImageMetadata(fname)
    exiv_image_metadata.read()
    exiv_image_metadata["Exif.Image.ImageDescription"] = desc
    exiv_image_metadata.write()

def getImageMetadata(fname):
    exiv_image_metadata = pyexiv2.ImageMetadata(fname)
    exiv_image_metadata.read()
    desc = exiv_image_metadata["Exif.Image.ImageDescription"]
    datetimeoriginal= exiv_image_metadata["Exif.Photo.DateTimeOriginal"]
    return (desc, datetimeoriginal)


# get GPS information from image file
def gpsInfo(img):
    gps = ()
    try:
        # Get the data from image file and return a dictionary
        data = gpsphoto.getGPSData(img)

        if "Latitude" in data and "Longitude" in data:
            gps = (round(data["Latitude"],6), round(data["Longitude"], 6))
    except Exception as e:
        st.error(f'exception occurred in extracting lat/ lon data: {e}')
    return gps


def setGpsInfo(fn, lat, lon):
    photo = gpsphoto.GPSPhoto(fn)
    info = gpsphoto.GPSInfo((round(float(lat), 6), round(float(lon), 6)))
    photo.modGPSData(info, fn)

# get timestamp from image file
def getTimestamp(img):
    print(f"-> {img}")
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
# def getDateTime(img):
#     value = []
#     # open the image
#     image = Image.open(img)

#     # extracting the exif metadata
#     exifdata = image.getexif()
#     date_time = exifdata.get(306)
#     if date_time:
#         value = (date_time.split(" ")[0]).split(":")[:3]
#         value.append(date_time)
#     else:
#         value = default_date_time
#     return value

# collect all metadata
def getMetadata(img):
    lat_lon = gpsInfo(img=img)
    desc, dt = getImageMetadata(img)
    if not desc and lat_lon:
        desc = getLocationDetails(lat_lon)
        if not desc:
            desc = def_name
    #res = getTimestamp(img)
    # res.append(lat_lon[0])
    # res.append(lat_lon[1])
    # res.append(getLocationDetails(lat_lon))
    # print(res)
    return (desc, lat_lon, dt)