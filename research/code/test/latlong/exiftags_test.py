#!/bin/python
import os
import sys
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

'''
https://exiv2.org/tags.html
'''

image = '/home/madhekar/work/home-media-app/data/input-data-1/img/AnjaliBackup/bf98198d-fcc6-51fe-a36a-275c06005669/IMAG0191.jpg'


def get_exif(image_file_path):

    exif_table = {}

    image = Image.open(image_file_path)

    info = image.getexif()

    date_time = info.get(306)
    print(date_time)

    desc = info.get(1502)
    print(desc)

    for tag, value in info.items():

        #print(tag, value)
        decoded = TAGS.get(tag, tag)

        exif_table[decoded] = value
        #print(exif_table)

    gps_info = {}
    print(exif_table['GPSInfo'])
    for key in exif_table.keys():

        decode = GPSTAGS.get(key,key)

        gps_info[decode] = exif_table[key]

    return gps_info

get_exif(image)