from PIL import Image
import os

# im = Image.open("/home/madhekar/temp/IMG_8499.PNG")
# exif = im.getexif()
# exif[0x9286] = "test"
# im.save("/home/madhekar/temp/IMG_8499.PNG", exif=exif)

# def write_metadata(img, ipath):
#    exif = img._getexif()
#    exif[0x9286] = "scenic"
#    img.save(ipath, exif=exif)

def read_metadata(img, ipath):
    exif = img._getexif()
    if not exif:
        raise Exception(f"Image {ipath} does not have EXIF data.")
    
    stype = exif[0x9286].decode("utf-8")
    return exif[0x9286], exif[0x9286].decode("utf-8").replace("ASCII\x00\x00\x00",""), exif[36867]
    
if __name__=='__main__':
    ipath = "/home/madhekar/work/home-media-app/data/input-data/img/Samsung_USB/b6f657c7-7b7f-5415-82b7-e005846a6ef5"
    for p in os.listdir(ipath):
        im = Image.open(os.path.join(ipath,p))    

        #write_metadata(im, ipath)

        print(read_metadata(im, p))

