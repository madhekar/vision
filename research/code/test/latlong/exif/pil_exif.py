from PIL import Image

im = Image.open("/Users/emadhekar/Pictures/45068E44-D877-4E6D-8BC9-BC7698786267.jpeg")
exif = im.getexif()
exif[0x9286] = "test"
im.save("/Users/emadhekar/Pictures/45068E44-D877-4E6D-8BC9-BC7698786267.jpeg", exif=exif)