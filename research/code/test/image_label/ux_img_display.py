from collections import defaultdict
from pathlib import Path
import streamlit as st

# Example list of absolute image paths
image_paths = [
"/mnt/zmdata/home-media-app/data/input-data/img/KIDDO/637ab7e9-5298-53d6-b7c0-8911a01964b1/Image009.jpg",
"/mnt/zmdata/home-media-app/data/input-data/img/KIDDO/637ab7e9-5298-53d6-b7c0-8911a01964b1/Image031.jpg",
"/mnt/zmdata/home-media-app/data/input-data/img/KIDDO/637ab7e9-5298-53d6-b7c0-8911a01964b1/Image007.jpg",
"/mnt/zmdata/home-media-app/data/input-data/img/KIDDO/637ab7e9-5298-53d6-b7c0-8911a01964b1/Image035.jpg",
"/mnt/zmdata/home-media-app/data/input-data/img/KIDDO/20f0e993-b00a-5e20-80a2-e1793273d921/CIMG1905.JPG",
"/mnt/zmdata/home-media-app/data/input-data/img/KIDDO/20f0e993-b00a-5e20-80a2-e1793273d921/CIMG1904.JPG",
"/mnt/zmdata/home-media-app/data/input-data/img/KIDDO/259f9567-cb75-5962-9742-7db826de4c96/Image020.jpg",
"/mnt/zmdata/home-media-app/data/input-data/img/KIDDO/259f9567-cb75-5962-9742-7db826de4c96/Image023.jpg",
"/mnt/zmdata/home-media-app/data/input-data/img/KIDDO/259f9567-cb75-5962-9742-7db826de4c96/Image026.jpg",
"/mnt/zmdata/home-media-app/data/input-data/img/KIDDO/259f9567-cb75-5962-9742-7db826de4c96/Image011.jpg"
]

# 1. Group image paths by their immediate parent subfolder
grouped_images = defaultdict(list)
for path_str in image_paths:
    path = Path(path_str)
    subfolder_name = path.parent.name
    grouped_images[subfolder_name].append(path_str)

# 2. Display each group with a header, image grid, and divider
subfolders = list(grouped_images.items())

for idx, (subfolder, paths) in enumerate(subfolders):
    st.subheader(f"📁 {subfolder}")

    # Optional: Display images in a 3-column grid instead of a single vertical list
    cols = st.columns(3)
    for i, img_path in enumerate(paths):
        with cols[i % 3]:
            st.image(img_path, caption=Path(img_path).name)#, use_container_width=True)

    # Add a divider between subfolders (skipping the very last one)
    if idx < len(subfolders) - 1:
        st.divider()
