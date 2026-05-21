from PIL import Image

try:
    img = Image.open("IMG_0241.PNG")
    img.save("fixed.png")
except Exception as e:
    print(f"Error: {e}")