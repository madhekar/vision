import streamlit as st
import datetime

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
st.title( "Media Search Portal")
st.logo("/home/madhekar/Pictures/IMGP3290.JPG")

image, video, documents = st.tabs(["***Image***", "***Video***", "***Documents***"])

# load data
cImgs, cTxts = init()

if "document" not in st.session_state:
    st.session_state["document"] = st.empty()

if "timgs" not in st.session_state:
    st.session_state["timgs"] = []
    
if "meta" not in st.session_state:
    st.session_state["meta"] = []

if "llm_text" not in st.session_state:
    st.session_state["llm_text"] = st.empty()

if "imgs" not in st.session_state:
    st.session_state["imgs"] = []

st.sidebar.title("seach criteria")
# with st.sidebar.form(key='attributes'):

ms = st.sidebar.multiselect( "select modalities for search result ", ["image", "text", "video", "audio"], ["image", "text"])

s = st.sidebar.selectbox("select search modality", ("text", "image"), index=1)

sim = st.sidebar.file_uploader("search doc/image: ", type=["png", "jpeg", "mpg", 'jpg','PNG','JPG'])

im = st.empty()
if sim:
    im = Image.open(sim)
    name = sim.name
    st.sidebar.image(im, caption="selected image")
    #st.sidebar.write(st.session_state["llm_text"])
    with open(name, "wb") as f:
        f.write(sim.getbuffer())

modalityTxt = st.sidebar.text_input(
    "search images based on following",
    placeholder="search images for...",
    disabled=False
)

dr = st.sidebar.date_input("select date range", datetime.date(2022,1,1))

search_btn = st.sidebar.button(label="Search")

if search_btn:
    # create query on image, also shows similar document in vector database (not using LLM)  -- openclip embedding function!
    embedding_function = OpenCLIPEmbeddingFunction()

    st.session_state["document"] = cTxts.query(
        query_embeddings=embedding_function("./" + sim.name),
        n_results=1,
    )["documents"][0][0]

    if "timgs" in st.session_state:
        st.session_state["timgs"] = []
        
    if "meta" in st.session_state:
        st.session_state["meta"] = []   
    
    qmdata = util.getMetadata(sim.name)

    st.session_state["imgs"] = cImgs.query(
        query_uris="./" + sim.name,
        include=["data", "metadatas"],
        n_results=6,
        where={
            "$and": [{"year": qmdata[0]}, {"month": qmdata[1]}, {"day": qmdata[2]}]
        },
    )

    for img in st.session_state["imgs"]["data"][0][1:]:
        st.session_state["timgs"].append(img)
    for mdata in st.session_state["imgs"]["metadatas"][0][1:]:
        st.session_state["meta"].append("Desc:[" + mdata.get("description") +"] ) People: ["+ mdata.get("names") + "] Location: [" + mdata.get("location") + "] Date: [" + mdata.get("datetime") + "]")


# Image TAB
with image:
  st.header("Similar Images")
  if st.session_state["timgs"] and len(st.session_state["timgs"]) > 1:
    
    index = image_select(
        label=">",
        images=st.session_state["timgs"],
        use_container_width=True,
        #captions=st.session_state["meta"],
        index=0,
        return_value='index'
    )

    im = Image.fromarray(st.session_state["timgs"][index])
    nim = ImageOps.expand(im,border=(2,2,2,2), fill=(222,222,222))
    c1,c2 = st.columns([9, 1])
    display_im = c1.image(nim, use_column_width="always")
    c2.write(''' **Description** ''')
    c2.write(st.session_state["imgs"]["metadatas"][0][1:][index]["description"])
    c2.write(''' **People/ Names** ''')
    c2.write(st.session_state["imgs"]["metadatas"][0][1:][index]["names"])
    c2.write(''' **Location** ''')
    c2.write(st.session_state["imgs"]["metadatas"][0][1:][index]["location"])
    c2.write(''' **Date** ''')
    c2.write(st.session_state["imgs"]["metadatas"][0][1:][index]["datetime"])
    #rot = nim.rotate(-90)
    #display_im.image(rot)
  else:
      st.write("sorry, no similar images found in search criteria!")  


#  Video TAB
with video:
    st.header("Similar Videos")
    st.write("sorry, no similar videos found in search criteria!")

# Documents Tab

with documents:
    st.header("Similar Documents")
    st.write("sorry, no similar documents found in search criteria!")
