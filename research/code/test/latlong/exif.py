from PIL import Image
from PIL.Image import Exif
from PIL.ExifTags import TAGS, GPSTAGS

def get_exif(file_name) -> Exif:
    image: Image.Image = Image.open('/home/madhekar/work/home-media-app/data/input-data/img/Samsung_USB/b6f657c7-7b7f-5415-82b7-e005846a6ef5/5d1fbc96-849b-4716-a835-2615ddb21830.jpg')
    return image.getexif()


def get_geo(exif):
    for key, value in TAGS.items():
        if value == "GPSInfo":
            break

        gps_info = exif.get_ifd(key)

        return {
           GPSTAGS.get(key, key): value
           for key, value in gps_info.items()
        }

if __name__ == '__main__':
    exif = get_exif("Tests/images/exif_gps.jpg")
    print(exif)
    geo = get_geo(exif)
    print(geo)