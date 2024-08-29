import streamlit as st
import datetime
from dateutil import parser
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# from streamlit_option_menu import option_menu
from streamlit_image_select import image_select
from PIL import Image, ImageOps
import util
import streamlit_init
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from loadData import init

streamlit_init.initUI()

# load data
cImgs, cTxts = init()

st.title( "Home Media Portal")

st.logo("/home/madhekar/work/zsource/zesha-high-resolution-logo.jpeg")

image, video, text = st.tabs(["Image", "Video", "Text"])



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

if "dt_range" not in st.session_state:
    st.session_state["dt_range"] = (datetime.datetime(2010,1,1), datetime.datetime(2019,1,1))   

with st.sidebar:
    st.header(":blue[Seach Criteria]")

    st.divider()

    s = st.selectbox(label="## Search type", options=("text", "image"), index=1)

    ms = st.multiselect(
        label="## Result types",
        options=["image", "text", "video", "audio"],
        default=["image", "text"],
    )

    if s == "image":
        sim = st.file_uploader(
            label="## Select Image",label_visibility="hidden", type=["png", "jpeg", "mpg", "jpg", "PNG", "JPG"]
        )
        im = st.empty()
        if sim:
            im = Image.open(sim)
            name = sim.name
            st.sidebar.image(im, caption="")
            # st.sidebar.write(st.session_state["llm_text"])
            with open(name, "wb") as f:
                f.write(sim.getbuffer())
    elif s == "text":
        modalityTxt = st.text_input(
            label="## Search text",
            placeholder="search modality types for...",
            disabled=False,
        )

    # dr = st.sidebar.date_input("** :blue[select date range]**", datetime.date(2022,1,1))

    #st.markdown("**:blue[select date range]**")


    def date_change():
        st.session_state["dt_range"] = st.session_state.mySlider
 
    date_range = st.slider(
        label="## Date range",
        key="mySlider",
        value=st.session_state["dt_range"],
        min_value=streamlit_init.MIN_DT,
        max_value=streamlit_init.MAX_DT,
        step=datetime.timedelta(days=1),
        on_change=date_change,
    )   

    search_btn = st.button(label="## **:blue[Search]**")

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
        #dt_format = "%Y-%m-%d %H:%M:%S"
        #st.write(qmdata[3])
        d = parser.parse(qmdata[3]).timestamp()
        #st.write(
        #    d,
        #    st.session_state["dt_range"][0].timestamp(),
        #    st.session_state["dt_range"][1].timestamp(),
        #)
        
        # execute image query with search criteria
        st.session_state["imgs"] = cImgs.query(
            query_uris="./" + sim.name,
            include=["data", "metadatas"],
            n_results=6,
        )

        #st.write(st.session_state["imgs"])

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

        #c2.divider()
        col21,col22,col23 = c2.columns([1,1,1], gap='small')
        with col21:
            right = st.button(label="## **:blue[&#x27A1;]**")
        with col22:
            left = st.button(label="## **:blue[&#x2B05;]**")
        with col23:
            flip = st.button(label="## **:blue[&#x2195;]**")

        imageLoc = c1.empty()
        display_im = imageLoc.image(nim, use_column_width="always")
        #st.button(st.image(nim, use_column_width="always"))

        if right:
            nim = nim.rotate(-90)
            imageLoc.image(nim, use_column_width="always")
        if left:
            nim = nim.rotate(90)
            imageLoc.image(nim, use_column_width="always")
        if flip:
            nim = nim.rotate(180)
            imageLoc.image(nim, use_column_width="always")
  
        #c2.divider()
        colt,cole = c2.columns([.7,.3])
        with colt:
           st.markdown("<p class='big-font-subh'>Gleeful Desc</p>", unsafe_allow_html=True)
        with cole:
                edit = st.button(label="## **:blue[&#x270D;]**")  

        if edit:
            util.update_metadata(
                id=st.session_state["imgs"]["ids"][0][index],
                desc=st.session_state["imgs"]["metadatas"][0][1:][index]["description"],
                names=st.session_state["imgs"]["metadatas"][0][1:][index]["names"],
                dt=st.session_state["imgs"]["metadatas"][0][1:][index]["datetime"],
                loc=st.session_state["imgs"]["metadatas"][0][1:][index]["location"]
            ) 
        o_desc = f'<p class="big-font">{st.session_state["imgs"]["metadatas"][0][1:][index]["description"]}</p>'  
        c2.markdown(o_desc, unsafe_allow_html=True)

        c2.write("<p class='big-font-subh'>People</p>", unsafe_allow_html=True)
        o_names = f'<p class="big-font">{st.session_state["imgs"]["metadatas"][0][1:][index]["names"]}</p>'
        c2.markdown(o_names, unsafe_allow_html=True)

        c2.write("<p class='big-font-subh'>Date Time</p>", unsafe_allow_html=True)
        o_datetime = f'<p class="big-font">{st.session_state["imgs"]["metadatas"][0][1:][index]["datetime"]}</p>'
        c2.markdown(o_datetime, unsafe_allow_html=True)

        c2.write("<p class='big-font-subh'>Location</p>",unsafe_allow_html=True)
        o_location = f'<p class="big-font">{st.session_state["imgs"]["metadatas"][0][1:][index]["location"]}</p>'
        c2.markdown(o_location, unsafe_allow_html=True)

        geolocator = Nominatim(user_agent="Z lookup")
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
        location = geolocator.geocode(st.session_state["imgs"]["metadatas"][0][1:][index]["location"])
        lat = location.latitude
        lon = location.longitude
        map_data = pd.DataFrame({'lat': [lat], 'lon': [lon]})
        c2.markdown("<p class='big-font-subh'>Map</p>",unsafe_allow_html=True)
        c2.map( map_data, zoom=12, size=100, color='#ff00ff')


    else:

        st.write("<p class='big-font'>sorry, no similar images found in search criteria!</p>", unsafe_allow_html=True)  


#  Video TAB
with video:
    #st.header("Similar Videos")
    st.write("<p class='big-font'>sorry, no similar videos found in search criteria!</p>", unsafe_allow_html=True)

#  Documents Tab
with text:
    if  st.session_state["document"] and len(st.session_state["document"]) > 1:
        st.text_area(label="Related text", value=st.session_state["document"])
    else:
        st.write("<p class='big-font'>sorry, no similar documents found in search criteria!</p>", unsafe_allow_html=True)


