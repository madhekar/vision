
import os
import base64
import hashlib
import glob
from PIL import Image
import pyexiv2
from GPSPhoto import gpsphoto
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import datetime
from dateutil import parser
import streamlit as st

default_home_loc = (32.968699774829794, -117.18420145463236)
default_date_time = ['2000','01','01','2000:01:01 01:01:01'] 
def_date_time = '2000:01:01 01:01:01'

names = ["Esha", "Anjali", "Bhalchandra"]
subclasses = [
    "Esha",
    "Anjali",
    "Bhalchandra",
    "Esha,Anjali",
    "Esha,Bhalchandra",
    "Anjali,Bhalchandra",
    "Esha,Anjali,Bhalchandra",
    "Bhalchandra,Sham",
    "Esha,Aaji",
    "Esha,Kumar",
    "Aaji",
    "Kumar",
    "Esha,Anjali,Shibangi",
    "Esha,Shibangi",
    "Anjali,Shoma",
    "Shibangi",
    "Shoma",
    "Bhiman",
]
# recursive call to get all image filenames
def getRecursive(rootDir, chunk_size=10):
    f_list=[]
    for fn in glob.iglob(rootDir + '/**/*', recursive=True):
        if not os.path.isdir(os.path.abspath(fn)):
            f_list.append(os.path.abspath(fn))
    for i in range(0, len(f_list), chunk_size):
        yield f_list[i:i+chunk_size]        
      

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

# get timestamp from image file
def getTimestamp(img):
    value = ""
    image = Image.open(img)
    # extracting the exif metadata
    exifdata = image.getexif()
    date_time = exifdata.get(306)
    #print(date_time)
    if date_time:
        date_time = str(date_time).replace('-',':')
        value = datetime.datetime.timestamp(datetime.datetime.strptime( date_time, "%Y:%m:%d %H:%M:%S"))
    else:
        value= datetime.datetime.timestamp(datetime.datetime.strptime(def_date_time, "%Y:%m:%d %H:%M:%S")
            )
        
    return value    

# get GPS information from image file
def gpsInfo(img):
    gps = ()
    # Get the data from image file and return a dictionary
    data = gpsphoto.getGPSData(img)
    #print(data)
    if 'Latitude' in data and 'Longitude' in data:
        gps = (data["Latitude"], data["Longitude"])
    else:
        gps = default_home_loc
    return gps

# get location address information from latitude and longitude
def getLocationDetails(strLnL):
    address = "n/a"

    geolocator = Nominatim(user_agent="zs")
    
    rev = RateLimiter(geolocator.reverse, min_delay_seconds=1)
    
    location = rev(strLnL, language='en', exactly_one = True)
    if location:
        address = location.address
    return address

# collect all metadata
def getMetadata(img):
    res = getTimestamp(img)
    lat_lon = gpsInfo(img=img)
    res.append(lat_lon[0])
    res.append(lat_lon[1])
    res.append(getLocationDetails(lat_lon))
    print(res)
    return res

def to_deg(value, loc):
        if value < 0:
            loc_value = loc[0]
        elif value > 0:
            loc_value = loc[1]
        else:
            loc_value = ""
        abs_value = abs(value)
        deg =  int(abs_value)
        t1 = (abs_value-deg)*60
        min = int(t1)
        sec = round((t1 - min)* 60, 5)
        return (deg, min, sec, loc_value) 
    
def setGpsLocation(fname, lat, lng):
    
    lat_deg = to_deg(lat, ["S", "N"])
    lng_deg = to_deg(lng, ["W", "E"])

    print ('lat:', lat_deg, ' lng:', lng_deg)

    # convert decimal coordinates into degrees, minutes and seconds
    exiv_lat = (pyexiv2.Rational(lat_deg[0]*60+lat_deg[1],60),pyexiv2.Rational(lat_deg[2]*100,6000), pyexiv2.Rational(0, 1))
    exiv_lng = (pyexiv2.Rational(lng_deg[0]*60+lng_deg[1],60),pyexiv2.Rational(lng_deg[2]*100,6000), pyexiv2.Rational(0, 1))

    exiv_image = pyexiv2.Image(fname)
    exiv_image.readMetadata()
    exif_keys = exiv_image.exifKeys() 
    print('exif keys: ', exif_keys)

    exiv_image["Exif.GPSInfo.GPSLatitude"] = exiv_lat
    exiv_image["Exif.GPSInfo.GPSLatitudeRef"] = lat_deg[3]
    exiv_image["Exif.GPSInfo.GPSLongitude"] = exiv_lng
    exiv_image["Exif.GPSInfo.GPSLongitudeRef"] = lng_deg[3]
    exiv_image["Exif.Image.GPSTag"] = 654
    exiv_image["Exif.GPSInfo.GPSMapDatum"] = "WGS-84"
    exiv_image["Exif.GPSInfo.GPSVersionID"] = '2 0 0 0'

    exiv_image.writeMetadata()


def img_to_base64bytes(img_path):
    with open(img_path, "rb") as f:
        data = f.read()
        return base64.b64encode(data).decode()

def generate_sha256_hash(txt):
    sha256_hash = hashlib.sha256()
    sha256_hash.update(txt.encode('utf-8'))
    return sha256_hash.hexdigest()

@st.dialog("Update Image Metadata")
def update_metadata(id, desc, names, dt, loc):
    _id = id
    print(_id)
    st.text_input(label="description:", value=desc)
    st.text_input(label="names", value=names)
    st.text_input(label="datetime", value=dt)
    st.text_input(label="location", value=loc)

    if st.button("Submit"):
        #st.session_state.vote = {"item": item, "reason": reason}
      st.rerun()
#iqn phase mono crystalline