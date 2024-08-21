import streamlit as st
import datetime
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# from streamlit_option_menu import option_menu
from streamlit_image_select import image_select
from PIL import Image, ImageOps
import util
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from loadData import init

st.set_page_config(
    page_title="zesha: Multi Modality Search (MMS)",
    page_icon="",
    initial_sidebar_state="auto",
    layout="wide",
)  # (margins_css)

st.title( "Home Media Portal")

st.html("""
        <style>
            [alt=Logo] {
            height: 6rem;
        }
        </style>
        """)
st.logo("/home/madhekar/work/zsource/zesha-high-resolution-logo.jpeg")

image, video, text = st.tabs(["**Image**", "**Video**", "**Text**"])

# load data
cImgs, cTxts = init()

if "document" not in st.session_state:
    st.session_state["document"] = []

if "timgs" not in st.session_state:
    st.session_state["timgs"] = []
    
if "meta" not in st.session_state:
    st.session_state["meta"] = []

if "llm_text" not in st.session_state:
    st.session_state["llm_text"] = st.empty()

if "imgs" not in st.session_state:
    st.session_state["imgs"] = []

st.sidebar.header("seach criteria")
# with st.sidebar.form(key='attributes'):

st.sidebar.divider()

s = st.sidebar.selectbox("select search modality type", ("text", "image"), index=1)

st.sidebar.divider()

ms = st.sidebar.multiselect( "select result modality types", ["image", "text", "video", "audio"], ["image", "text"])

st.sidebar.divider()

if s == "image":
  sim = st.sidebar.file_uploader("search image ", type=["png", "jpeg", "mpg", 'jpg','PNG','JPG'])
  im = st.empty()
  if sim:
    im = Image.open(sim)
    name = sim.name
    st.sidebar.image(im, caption="selected image")
    #st.sidebar.write(st.session_state["llm_text"])
    with open(name, "wb") as f:
        f.write(sim.getbuffer())
elif s == "text":
  modalityTxt = st.sidebar.text_input(
    "search types based on text",
    placeholder="search modality types for...",
    disabled=False
  )

st.sidebar.divider()

dr = st.sidebar.date_input("select date range", datetime.date(2022,1,1))

st.sidebar.divider()

search_btn = st.sidebar.button(label="Search")

#seach button pressed
if search_btn:
    # create query on image, also shows similar document in vector database (not using LLM)  -- openclip embedding function!
    embedding_function = OpenCLIPEmbeddingFunction()

    # reset session_state for temperory images for view
    if "timgs" in st.session_state:
        st.session_state["timgs"] = []
        
    # reset session_state for metadata    
    if "meta" in st.session_state:
        st.session_state["meta"] = []   
    
    if s == "image":
      # execute text collection query
      st.session_state["document"] = cTxts.query(
        query_embeddings=embedding_function("./" + sim.name),
        n_results=1,
      )["documents"][0][0]

      # get location and datetime metadata for an image
      qmdata = util.getMetadata(sim.name)

      # execute image query with search criteria
      st.session_state["imgs"] = cImgs.query(
        query_uris="./" + sim.name,
        include=["data", "metadatas"],
        n_results=6,
        where={
            "$and": [{"year": qmdata[0]}, {"month": qmdata[1]}, {"day": qmdata[2]}]
        },    
       )        
      st.write(st.session_state["imgs"])

    elif s == "text":
        # execute text collection query
        st.session_state["document"] = cTxts.query(
          query_texts=modalityTxt,
          n_results=1,
        )["documents"][0][0]

        # execute image query with search criteria
        st.session_state["imgs"] = cImgs.query(
            query_texts=modalityTxt,
            include=["data", "metadatas"],
            n_results=6
        )   

    for img in st.session_state["imgs"]["data"][0][1:]:
        st.session_state["timgs"].append(img)
    for mdata in st.session_state["imgs"]["metadatas"][0][1:]:
        st.session_state["meta"].append("Desc:[" + mdata.get("description") +"] ) People: ["+ mdata.get("names") + "] Location: [" + mdata.get("location") + "] Date: [" + mdata.get("datetime") + "]")


# Image TAB
with image:
    #st.subheader("Similar Images")
    if st.session_state["timgs"] and len(st.session_state["timgs"]) > 1:
        index = image_select(
            label="Similar Images",
            images=st.session_state["timgs"],
            use_container_width=True,
            # captions=st.session_state["meta"],
            index=0,
            return_value="index",
        )

        im = Image.fromarray(st.session_state["timgs"][index])
        nim = ImageOps.expand(im, border=(2, 2, 2, 2), fill=(200, 200, 200))
        c1, c2 = st.columns([9, 1])
        imageLoc = c1.empty()
        display_im = imageLoc.image(nim, use_column_width="always")
        c2.divider()
        c2.markdown(" **:blue[Description]** ")
        c2.write(st.session_state["imgs"]["metadatas"][0][1:][index]["description"])
        c2.markdown("**:blue[People/ Names]**")
        c2.write(st.session_state["imgs"]["metadatas"][0][1:][index]["names"])
        c2.markdown(" **:blue[DateTime]**")
        c2.write(st.session_state["imgs"]["metadatas"][0][1:][index]["datetime"])
        c2.markdown(" **:blue[Location]** ")
        c2.write(st.session_state["imgs"]["metadatas"][0][1:][index]["location"])

        geolocator = Nominatim(user_agent="Z lookup")
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
        location = geolocator.geocode(st.session_state["imgs"]["metadatas"][0][1:][index]["location"])
        lat = location.latitude
        lon = location.longitude
        map_data = pd.DataFrame({'lat': [lat], 'lon': [lon]})
        c2.markdown(" **:blue[Map]** ")
        c2.map( map_data, zoom=12, size=2)

        nim = nim.rotate(-90)
        imageLoc.image(nim, use_column_width="always")
    else:
        st.write("sorry, no similar images found in search criteria!")  


#  Video TAB
with video:
    st.header("Similar Videos")
    st.write("sorry, no similar videos found in search criteria!")

#  Documents Tab
with text:
    if  st.session_state["document"] and len(st.session_state["document"]) > 1:
        st.text_area("related text", value=st.session_state["document"])
    else:      
      st.write("sorry, no similar documents found in search criteria!")
