from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# try:
#     img = Image.open("IMG_0241.PNG")
#     img.save("fixed.png")
# except Exception as e:
#     print(f"Error: {e}")


try:
    with Image.open("IMG_0241.PNG") as img:
        img.verify()
    print("Image is valid.")
except Exception as e:
    print(f"Image is corrupted: {e}")