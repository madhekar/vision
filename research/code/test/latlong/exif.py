from PIL import Image
from PIL.Image import Exif
from PIL.ExifTags import TAGS, GPSTAGS

def get_exif(file_name) -> Exif:
    image: Image.Image = Image.open(file_name)
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
    geo = get_geo(exif)
    print(geo)