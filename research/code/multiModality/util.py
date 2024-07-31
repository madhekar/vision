import sys
from PIL import Image
from PIL.ExifTags import TAGS
import pyexiv2
import fractions
from GPSPhoto import gpsphoto
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

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
    #print(data)
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

    # convert decimal coordinates into degrees, munutes and seconds
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

