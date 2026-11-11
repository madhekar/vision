import streamlit as st

st.title("Streamlit Image Grid with Custom Border Colors")

# List of image URLs or local paths
images = [
    {"url": "https://picsum.photos", "color": "red"},
    {"url": "https://picsum.photos", "color": "blue"},
    {"url": "https://picsum.photos", "color": "green"},
    {"url": "https://picsum.photos", "color": "orange"},
]

# Create a 2-column grid
cols_per_row = 2
rows = [images[i : i + cols_per_row] for i in range(0, len(images), cols_per_row)]

for row in rows:
  cols = st.columns(len(row))
  for idx, item in enumerate(row):
    with cols[idx]:
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
                    <img src="{item['url']}" style="width: 100%; border-radius: 4px; display: block;">
                </div>
                """,
          unsafe_allow_html=True,
      )
