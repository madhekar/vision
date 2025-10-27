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
from PIL import Image, ExifTags
import streamlit as st
from math import cos,asin,sqrt,radians,sin
from GPSPhoto import gpsphoto
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from utils.util import ball_tree as bt

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
cache = {}
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

def format_lat_lon(df):
    lat_lon = ['GPSLatitude','GPSLongitude'] 
    df[lat_lon] = df[lat_lon].applymap(lambda x: str(round(float(x) ,6)) if not x == '-' else x )
    print(f'transformed: {df.head()}')
    return df

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


def get_image_exif_info(image_path):
    user_comment, datetime_original = "",""
    try:
        img = Image.open(image_path)
        exif_data = img.getexif()

        decoded_exif = {}
        for tag_id, value in exif_data.items():
            tag_name = ExifTags.TAGS.get(tag_id, tag_id)
            decoded_exif[tag_name] = value

        user_comment = decoded_exif.get("UserComment")
        datetime_original = decoded_exif.get("DateTimeOriginal")

        print(f"Image: {image_path}")
        print(f"User Comment: {user_comment}")
        print(f"Original Date and Time: {datetime_original}")

    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return datetime_original, user_comment    

# get timestamp from image file
"""
def read_metadata(img, ipath):
    exif = img._getexif()
    if not exif:
        raise Exception(f"Image {ipath} does not have EXIF data.")
    
    stype = exif[0x9286].decode("utf-8")
    return exif[0x9286], exif[0x9286].decode("utf-8").replace("ASCII\x00\x00\x00",""), exif[36867]
"""
def getTimestamp(img):
    print(f"-> {img}")
    value = ""
    user_comment = ""
    image = Image.open(img)
    # extracting the exif metadata
    exifdata = image._getexif()
    print(f"-->{exifdata}")
    if exifdata:
        date_time = exifdata[36867]
        user_comment = exifdata[0x9286]
        if date_time:
            date_time = str(date_time).replace("-", ":")
            value = datetime.datetime.timestamp(
                datetime.datetime.strptime(date_time, "%Y:%m:%d %H:%M:%S")
            )
        else:
            value = datetime.datetime.timestamp(
                datetime.datetime.strptime(def_date_time, "%Y:%m:%d %H:%M:%S")
            )
        if user_comment:
            s_user_comment =  user_comment.decode('utf-8').replace("ASCII\x00\x00\x00","")   

    return value, s_user_comment

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

def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295
    haversine = (
        0.5
        - cos((lat2 - lat1) * p) / 2
        + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
    )
    return 12742 * asin(sqrt(haversine))

def closest(data, v):
    return min(data, key=lambda x: distance(v[0], v[1], x[0], x[1]))



    
