import piexif
from PIL import Image

img = Image.open(
    "/home/madhekar/work/home-media-app/data/input-data-1/img/AnjaliBackup/bf98198d-fcc6-51fe-a36a-275c06005669/IMAG0191.jpg"
)
exif_dict = piexif.load(img.info['exif'])

altitude = exif_dict['GPS'][piexif.GPSIFD.GPSAltitude]
latitude = exif_dict["GPS"][piexif.GPSIFD.GPSLatitude]
longitude = exif_dict["GPS"][piexif.GPSIFD.GPSLongitude]
print(altitude, latitude, longitude)


desc = exif_dict["0th"][piexif.ImageIFD.ImageDescription]

print(desc)