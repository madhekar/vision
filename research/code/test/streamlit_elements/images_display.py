import streamlit as st
import time
import os


def get_image_files(folder_path):
    """
    Returns a list of image filenames from a specified folder.
    """
    image_files = []
    valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']

    for filename in os.listdir(folder_path):
        # Get the file extension
        _, ext = os.path.splitext(filename)
        # Check if the extension is in the list of valid image extensions
        if ext.lower() in valid_extensions:
            filename = os.path.join(folder_path, filename)
            image_files.append(filename)
    return image_files



images = get_image_files("/home/madhekar/temp/travel/world/images")
print(images)

image_holder = st.empty()
for item in images:
    image_holder.image(item)
    time.sleep(1)