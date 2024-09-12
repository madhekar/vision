import streamlit as st
from st_click_detector import click_detector
import base64
from PIL import Image
from io import BytesIO
from io import StringIO


im = "/home/madhekar/work/zsource/zesha-high-resolution-logo.jpeg" 

def encode_img(img_fn):
    with open(img_fn, "rb") as f:
        data = f.read()
        return base64.b64encode(data).decode()
    
content = f"<a href="#" id="Image 1"><img  width="70%" src="data:image/png;base64, {encode_img(im)}" alt="zesha image"></a>"

clicked = click_detector(content)

if clicked:
    print(clicked)
    img = Image.open(im)
    img = img.rotate(-90)
    #by = img.tobytes()

    by = BytesIO()
    img.save(by, format="png")
    by.seek(0)
    bstr = base64.b64encode(by.getbuffer()).decode()
    #Image.open(bstr)
    #bstr = base64.b64encode(img.tobytes()).decode()
    f"<a href="#" id="Image 2"><img  width="70%" src="data:image/png;base64, {bstr}" alt="zesha image"></a>"

st.markdown(
    f"**{clicked} clicked**" if clicked != {encode_img(im)} else "**No click**",
    unsafe_allow_html=True,
)

#<p><a href="#" id="Link 1">First link</a></p>
# <img src="data:image/png;base64,{encoded_string}" alt="logo image" style="width:500px;height:500px;">