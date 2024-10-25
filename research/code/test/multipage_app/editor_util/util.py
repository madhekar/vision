
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
import pprint
import yaml
import pyexiv2
import streamlit as st
from GPSPhoto import gpsphoto
import streamlit_pydantic as sp
from pydantic import BaseModel

@st.cache_resource
def config_load():
    with open("editor_util/metadata_conf.yaml") as prop:
        dict = yaml.safe_load(prop)

        print("* * * * * * * * * * Metadata Generator Properties * * * * * * * * * * *")
        pprint.pprint(dict)
        print("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *")
        static_metadata_path = dict["metadata"]["static_metadata_path"]
        static_metadata_file = dict["metadata"]["static_metadata_file"]
        missing_metadata_path = dict["metadata"]["missing_metadata_path"]
        missing_metadata_file = dict["metadata"]["missing_metadata_file"]
    return (
        static_metadata_path,
        static_metadata_file,
        missing_metadata_path,
        missing_metadata_file,
    )

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


def setDateTimeOriginal(fname, dt):
    exiv_image = pyexiv2.Image(filename=fname)
    exiv_image["Exif"][pyexiv2.ExifIFD.DateTimeOriginal] = dt
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
    info = gpsphoto.GPSInfo((lat, lon))
    photo.modGPSData(info, fn)

class Location(BaseModel):
  locId: str
  desc: str
  lat: float
  lon: float