from collections import defaultdict
import math
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="Subfolder Image Gallery", layout="wide")
st.title("Image Gallery by Subfolder")

# 1. Your supplied list of absolute file paths
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

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}

# 2. Group the flat list by immediate parent folder name
@st.cache_data
def group_paths_by_subfolder(paths: list[str]):
    groups = defaultdict(list)
    for raw_path in sorted(paths):
        p = Path(raw_path)
        if p.suffix.lower() in VALID_EXTS:
            # p.parent.name extracts the immediate subfolder name (e.g., "cats")
            # Use str(p.parent) instead if different paths share the same leaf folder name
            groups[p.parent.name].append(str(p))
    return dict(groups)

grouped_images = group_paths_by_subfolder(image_paths)

if not grouped_images:
    st.warning("No valid image paths were provided.")
    st.stop()

# 3. Optional sidebar filter
selected_folder = st.sidebar.selectbox(
    "Filter by Subfolder",
    options=["All Subfolders"] + list(grouped_images.keys())
)

folders_to_show = (
    grouped_images.items()
    if selected_folder == "All Subfolders"
    else [(selected_folder, grouped_images[selected_folder])]
)

# 4. Grid settings
cols_per_row = 10
rows_per_page = 5
images_per_page = cols_per_row * rows_per_page

# 5. Render each subfolder group with pagination and columns
for folder_name, images in folders_to_show:
    st.subheader(f"📁 {folder_name} ({len(images)} images)")
    
    total_pages = max(1, math.ceil(len(images) / images_per_page))
    
    if total_pages > 1:
        page = st.number_input(
            f"Page ({folder_name})",
            min_value=1,
            max_value=total_pages,
            value=1,
            key=f"page_{folder_name}"
        )
    else:
        page = 1

    start_idx = (page - 1) * images_per_page
    end_idx = start_idx + images_per_page
    page_images = images[start_idx:end_idx]

    for row_start in range(0, len(page_images), cols_per_row):
        cols = st.columns(cols_per_row)
        row_images = page_images[row_start : row_start + cols_per_row]
        
        for col, img_path in zip(cols, row_images):
            with col:
                st.image(img_path, caption=Path(img_path).name) #, use_container_width=True)
                
    st.divider()
