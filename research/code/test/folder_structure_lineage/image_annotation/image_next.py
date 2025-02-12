import streamlit as st
from PIL import Image
import os

def image_slider(image_folder):
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        st.error("No images found in the specified folder.")
        return

    num_images = len(image_files)
    current_index = st.session_state.get('current_index', 0)

    image_path = os.path.join(image_folder, image_files[current_index])
    image = Image.open(image_path)
    st.image(image, use_column_width=True, caption=image_files[current_index])
    
    cols = st.columns([0.1, 0.8, 0.1])
    with cols[0]:
      if st.button("**Prev**", key="prev"):
          current_index = (current_index - 1) % num_images
          st.session_state['current_index'] = current_index
    with cols[2]:
      if st.button("**Next**", key="next"):
          current_index = (current_index + 1) % num_images
          st.session_state['current_index'] = current_index



if __name__ == '__main__':
    image_directory = "/home/madhekar/work/home-media-app/data/train-data/img" 
    image_slider(image_directory)