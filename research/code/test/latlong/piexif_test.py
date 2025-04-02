import piexif
from PIL import Image

img = Image.open(fname)
exif_dict = piexif.load(img.info['exif'])

altitude = exif_dict['GPS'][piexif.GPSIFD.GPSAltitude]
print(altitude)