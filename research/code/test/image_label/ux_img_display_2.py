import math
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="Subfolder Image Gallery", layout="wide")
st.title("Image Gallery by Subfolder")

# 1. Point this to your main directory containing subfolders
ROOT_DIR = Path("path/to/your/images")
VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}

# 2. Scan subfolders and collect image paths
@st.cache_data
def load_grouped_images(root_path: Path):
    groups = {}
    if not root_path.exists():
        return groups
    
    # Sort subfolders alphabetically
    for folder in sorted([p for p in root_path.iterdir() if p.is_dir()]):
        imgs = sorted([
            str(f) for f in folder.iterdir() 
            if f.is_file() and f.suffix.lower() in VALID_EXTS
        ])
        if imgs:
            groups[folder.name] = imgs
    return groups

grouped_images = load_grouped_images(ROOT_DIR)

if not grouped_images:
    st.warning(f"No subfolders with images found in `{ROOT_DIR}`.")
    st.stop()

# 3. Optional sidebar filter to view all subfolders or pick one
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
cols_per_row = 4
rows_per_page = 2
images_per_page = cols_per_row * rows_per_page

# 5. Render each subfolder with its own pagination and columns
for folder_name, images in folders_to_show:
    st.subheader(f"📁 {folder_name} ({len(images)} images)")
    
    total_pages = max(1, math.ceil(len(images) / images_per_page))
    
    # Unique key per subfolder prevents widget conflicts
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
                st.image(img_path, caption=Path(img_path).name, use_container_width=True)
                
    st.divider()
