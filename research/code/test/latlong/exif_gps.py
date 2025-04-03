from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

"""
https://exiv2.org/tags.html
"""

image = "/home/madhekar/work/home-media-app/data/input-data-1/img/AnjaliBackup/bf98198d-fcc6-51fe-a36a-275c06005669/IMAG0191.jpg"

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

print(get_gps_coordinates(image))