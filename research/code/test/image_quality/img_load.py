from PIL import Image
import io

def load_image(image_path):
    try:
        with open(image_path, 'rb') as img_bin:
            buff = io.BytesIO(img_bin.read())
            img = Image.open(buff)
            return img
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

# Example usage:
image = load_image("/Users/emadhekar/Downloads/1a5e9da6-462d-4f2f-a289-5c45e0db1176.JPG")
if image:
    # Process the image
    image.show()