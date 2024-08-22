import streamlit as st
from st_click_detector import click_detector
import base64

def encode_img(img_fn):
    with open(img_fn, "rb") as f:
        data = f.read()
        return base64.b64encode(data).decode()
    
content = f"<a href='#' id='Image 1'><img src='data:image/png;base64, {encode_img('/home/madhekar/work/zsource/zesha-high-resolution-logo.jpeg')}' alt='zesha image'></a>"

clicked = click_detector(content)

st.markdown(
    f"**{clicked} clicked**" if clicked != "" else "**No click**",
    unsafe_allow_html=True
)

#<p><a href='#' id='Link 1'>First link</a></p>
# <img src='data:image/png;base64,{encoded_string}' alt='logo image' style='width:500px;height:500px;'>