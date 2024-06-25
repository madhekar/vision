margins_css = """
   <style>
     .main > div {
       padding-left: 0rem;
       padding-right: 0rem;
     }
   </style>  
"""

import streamlit as st
from PIL import Image
from streamlit_image_select import image_select

st.set_page_config(layout='wide') #(margins_css)

st.title('MultiModality Search')

st.sidebar.multiselect('select search result modalities', ['img', 'txt', 'video', 'audio'])

search_img = st.sidebar.file_uploader('search doc/image: ', type='png')
img = None
if search_img:
  img = Image.open(search_img)
  st.sidebar.image(img, caption='search image')


#camera_img = st.sidebar.camera_input('take a camera picture') 


search_txt = st.sidebar.text_input('enter the search term: ', placeholder='enter shot search term', disabled=False)
st.sidebar.write('upload image:')
st.sidebar.button('submit!')

imgs = image_select(
   label='select image',
   images=[
      '/home/madhekar/work/zsource/family/img/IMG_8375.PNG',
      '/home/madhekar/work/zsource/family/img/IMG_6555.PNG',
      '/home/madhekar/work/zsource/family/img/IMG_6701.PNG',
      '/home/madhekar/work/zsource/family/img/IMG_7826.PNG',
      '/home/madhekar/work/zsource/family/img/IMG_6073.PNG'

   ],
   captions=['caption one','caption two', 'caption three','caption fore', 'caption five'],
)

st.image(imgs)