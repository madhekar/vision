import streamlit as st
import base64
from PIL import Image

def encode_img(img_fn):
    with open(img_fn, "rb") as f:
        data = f.read()
        return base64.b64encode(data).decode()

btn_html = """
	<a href="https://www.example.com" target="_blank" >
        <img src="/home/madhekar/Pictures/IMG_0974.jpg" alt="logo image" style="width:100px;height:100px;">
        </a>
"""
with open("/home/madhekar/Pictures/IMG_0070.JPG", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode()

# encoded =  base64.b64encode(Image.open('/home/madhekar/Pictures/IMG_0070.JPG'))
html= f"<a href='https://www.example.com'><img src='data:image/png;base64,{encode_img('/home/madhekar/Pictures/IMG_0070.JPG')}' alt='logo image' style='width:500px;height:500px;'></a>"

st.markdown(html, unsafe_allow_html=True)

