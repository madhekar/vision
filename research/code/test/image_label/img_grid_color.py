import streamlit as st
import base64
from pathlib import Path

st.title("Streamlit Image Grid with Custom Border Colors")

# List of image URLs or local paths



images = [
    {"url": "/home/madhekar/temp/filter/validation/people/IMG_8035.PNG", "color": "red"},
    {"url": "/home/madhekar/temp/filter/validation/people/IMG_9531.PNG", "color": "blue"},
    {"url": "/home/madhekar/temp/filter/validation/people/IMG_9530.PNG", "color": "green"},
    {"url": "/home/madhekar/temp/filter/validation/people/IMG_9445.PNG", "color": "orange"},
]


def get_base64_image(image_path):
  with open(image_path, "rb") as img_file:
    return base64.b64encode(img_file.read()).decode("utf-8")
  
# Create a 2-column grid
cols_per_row = 2
rows = [images[i : i + cols_per_row] for i in range(0, len(images), cols_per_row)]

for row in rows:
  cols = st.columns(len(row))
  for idx, item in enumerate(row):
    with cols[idx]:
      img_path = Path(item["url"])
      ext = img_path.suffix.lstrip(".").lower()
      if ext == "png":
        ext = "png"
      b64_data = get_base64_image(image_path=img_path)  
      # Render image with a colored border using inline CSS
      st.markdown(
          f"""
                <div style="
                    border: 4px solid {item['color']}; 
                    border-radius: 8px; 
                    padding: 4px; 
                    display: inline-block;
                    width: 100%;
                    box-sizing: border-box;
                ">
                    <img src="data:image/{ext};base64,{b64_data}" style="width: 100%; border-radius: 4px; display: block;">
                </div>
                """,
          unsafe_allow_html=True,
      )
