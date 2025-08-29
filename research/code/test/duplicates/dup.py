import hashlib
from PIL import Image
import imagehash
import os

def calculate_pHash(filepath):
    image_path = filepath
    image = Image.open(image_path)
    ph = imagehash.phash(image)
    return ph

def calculate_md5(filepath):
    with open(filepath, "rb") as f:
        file_hash = hashlib.md5()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest()

# Example usage:
image_path = "/Users/emadhekar/Downloads/1909ab0d-8449-47a4-848c-4a5d7c4832ad.JPG"
md5_hash = calculate_md5(image_path)
p_hash = calculate_pHash(image_path)
print(f"MD5 and pHash hash of {image_path}: {md5_hash} {p_hash}")