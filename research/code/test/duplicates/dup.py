import glob
import os
import hashlib
from PIL import Image
import imagehash
import os

def getRecursive(rootDir):
    f_list = []
    for fn in glob.iglob(rootDir + "/**/*", recursive=True):
        if not os.path.isdir(os.path.abspath(fn)):
            f_list.append(
                (
                    str(os.path.abspath(fn)).replace(str(os.path.basename(fn)), ""),
                    os.path.basename(fn),
                )
            )
    return f_list

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
print(f"MD5 and pHash hash of {image_path}: {md5_hash} pHash: {p_hash}")
