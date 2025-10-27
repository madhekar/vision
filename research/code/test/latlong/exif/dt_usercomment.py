from PIL import Image


def get_date_taken(path):
    exif = Image.open(path)._getexif()
    print(exif)

    for t, v in exif.items():
       print(f" tag: {t} value: {v}")
    if not exif:
        raise Exception("Image {0} does not have EXIF data.".format(path))
    return exif[36867]


print(get_date_taken("/home/madhekar/work/home-media-app/data/input-data/img/Samsung_USB/2a98fafb-a921-519f-8561-ed25ccd997de/fce7616e-d485-403b-ba29-e33d5b80df09.jpg"))