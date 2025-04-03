import math
from PIL import Image
from PIL.ExifTags import TAGS 
import pyexiv2
from fractions import Fraction 


image_file = "/home/madhekar/temp/exif/IMG_3197.jpg"

def_lat = 32.968699774829794
def_lon = -117.18420145463236

def deg_to_dms(deg, type="lat"):
    decimals, number = math.modf(deg)
    d = int(number)
    m = int(decimals * 60)
    s = (deg - d - m / 60) * 3600.00
    compass = {"lat": ("N", "S"), "lon": ("E", "W")}
    compass_str = compass[type][0 if d >= 0 else 1]
    return (abs(d), abs(m), abs(s), compass_str)


def get_decimal_from_dms(dms, ref):
    dec_degrees = dms[0] + dms[1] / 60.0 + dms[2] / 3600.0
    if ref in ["S", "W"]:
        dec_degrees *= -1
    return dec_degrees


def set_gps_location(file_name, lat, lon, desc="Bhal Test"):
    lat_dms = deg_to_dms(lat, type='lat')
    lon_dms = deg_to_dms(lon, type='lon')

    re= Fraction(33.0) + Fraction(45.0)
    print(re)
    print(lat_dms)
    exiv_image = pyexiv2.ImageMetadata(file_name)
    exiv_image.read()

    exiv_lat = (Fraction(Fraction(lat_dms[0])*Fraction(60.0)+Fraction(lat_dms[1]),Fraction(60.0)),Fraction(Fraction(lat_dms[2])*Fraction(100),Fraction(6000.0)), Fraction(Fraction(0), Fraction(1.0)))
    exiv_lng = (Fraction(Fraction(lon_dms[0])*Fraction(60.0)+Fraction(lon_dms[1]),Fraction(60.0)),Fraction(Fraction(lon_dms[2])*Fraction(100),Fraction(6000.0)), Fraction(Fraction(0), Fraction(1.0)))

    print(exiv_lat, exiv_lng)

    exif_keys = exiv_image.exif_keys

    v = float(f"{lat_dms[2]:.2f}")
    print(v)
    exiv_image["Exif.GPSInfo.GPSLatitude"] = (Fraction(lat_dms[0]), Fraction(lat_dms[1]), Fraction(v).limit_denominator())
    exiv_image['Exif.GPSInfo.GPSLatitudeRef'] = lat_dms[3]
    exiv_image['Exif.GPSInfo.GPSLongitude'] = exiv_lng
    exiv_image['Exif.GPSInfo.GPSLongitudeRef'] = lon_dms[3]
    exiv_image['Exif.Image.GPSTag'] = 654
    exiv_image["Exif.GPSInfo.GPSMapDatum"] = "WGS-84"
    exiv_image["Exif.GPSInfo.GPSVersionID"] = "2 0 0 0"

    exiv_image["Exif.Image.ImageDescription"] = desc

    exiv_image.write()



set_gps_location(image_file, def_lat, def_lon )