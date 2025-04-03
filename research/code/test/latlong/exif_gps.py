import math
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

"""
https://exiv2.org/tags.html
"""

image = "/home/madhekar/work/home-media-app/data/input-data-1/img/AnjaliBackup/bf98198d-fcc6-51fe-a36a-275c06005669/IMAG0191.jpg"

def deg_to_dms(deg, type='lat'):
        decimals, number = math.modf(deg)
        d = int(number)
        m = int(decimals * 60)
        s = (deg - d - m / 60) * 3600.00
        compass = {
            'lat': ('N','S'),
            'lon': ('E','W')
        }
        compass_str = compass[type][0 if d >= 0 else 1]
        return '{}º{}\'{:.2f}"{}'.format(abs(d), abs(m), abs(s), compass_str)

def get_decimal_from_dms(dms, ref):
    dec_degrees = dms[0] + dms[1] / 60.0 + dms[2] / 3600.0
    if ref in ['S','W']:
        dec_degrees *= -1
    return dec_degrees   

def get_gps_coordinates(file):

    img = Image.open(file)

    exif_data = img._getexif()

    if not exif_data or 34853 not in exif_data:
       print(None, None)

    gps_info = {}

    for key in exif_data[34853].keys():
        decode = GPSTAGS.get(key, key)
        gps_info[decode] = exif_data[34853][key]

    print(gps_info)
    gps_lat = gps_info['GPSLatitude']
    gps_lat_ref = gps_info["GPSLatitudeRef"]  
    gps_lon = gps_info["GPSLongitude"]
    gps_lon_ref = gps_info["GPSLongitudeRef"]

    print(f'{gps_lat}:{gps_lat_ref} -- {gps_lon}:{gps_lon_ref}')

    if gps_lat and gps_lat_ref and gps_lon and gps_lon_ref:

        lat = get_decimal_from_dms(gps_lat, gps_lat_ref)
        lon = get_decimal_from_dms(gps_lon, gps_lon_ref)  
        return lat, lon
    return None, None

lt,lo = get_gps_coordinates(image)
print(f'{lt} : {lo}')

dms_lat = deg_to_dms(lt, type='lat')
print(dms_lat)

dms_lon = deg_to_dms(lo, type='lon')
print(dms_lon)
