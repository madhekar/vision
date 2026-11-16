from collections import defaultdict
from pathlib import Path
import streamlit as st

# Example list of absolute image paths
image_paths = [
    "/path/to/root/cats/tabby.jpg",
    "/path/to/root/cats/siamese.png",
    "/path/to/root/dogs/golden.jpg",
    "/path/to/root/dogs/beagle.jpeg",
    "/path/to/root/birds/parrot.png",
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
            st.image(img_path, caption=Path(img_path).name, use_container_width=True)

    # Add a divider between subfolders (skipping the very last one)
    if idx < len(subfolders) - 1:
        st.divider()
