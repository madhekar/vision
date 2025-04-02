#!/bin/python
import os
import sys
from PIL import Image
from PIL.ExifTags import TAGS

image = '"/home/madhekar/work/home-media-app/data/input-data-1/img/AnjaliBackup/bf98198d-fcc6-51fe-a36a-275c06005669/IMAG0191.jpg'

for (tag,value) in Image.open(image)._getexif().iteritems():
        print (TAGS.get(tag), value)