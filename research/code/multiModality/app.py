import streamlit as st
import asyncio
import datetime

# from streamlit_option_menu import option_menu
from streamlit_image_select import image_select
from PIL import Image, ImageOps
import util


from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

from loadData import init
from LLM import setLLM, fetch_llm_text
# import chromadb as cdb

st.set_page_config(
    page_title="zesha: Multi Modality Search (MMS)",
    page_icon="",
    initial_sidebar_state="auto",
    layout="wide",
)  # (margins_css)
st.title("Zesha: MMS")

# load data
cImgs, cTxts = init()

# init LLM modules
m, t, p = setLLM()

if "document" not in st.session_state:
    st.session_state["document"] = ""

if "timgs" not in st.session_state:
    st.session_state["timgs"] = []
    
if "meta" not in st.session_state:
    st.session_state["meta"] = []

if "llm_text" not in st.session_state:
    st.session_state["llm_text"] = ""


def getLLMText(question, article):
    st.session_state['llm_text'] = fetch_llm_text(sim, model=m, processor=p, top=top, temperature=te, question = question, article=article)

st.sidebar.title("seach criteria")
# with st.sidebar.form(key='attributes'):

ms = st.sidebar.multiselect(
    "select search result modalities", ["img", "txt", "video", "audio"]
)

sim = st.sidebar.file_uploader("search doc/image: ", type=["png", "jpeg", "mpg", 'jpg','PNG','JPG'])

im = None
if sim:
    im = Image.open(sim)
    name = sim.name
    st.sidebar.image(im, caption="selected image to find simili...")
    with open(name, "wb") as f:
        f.write(sim.getbuffer())


txt = st.sidebar.text_input(
    "enter the search term: ", placeholder="enter search term: ", 
    #value= "Answer with organized answers: What type of flower is in the picture? Mention some of its characteristics and how to take care of it ?", 
    value= "Answer with organized answers: Please describe the entities in the picture, Also mention some facts about the picture.",
    disabled=False
)

dr = st.sidebar.date_input("select date range", datetime.date(2022,1,1))

top = st.sidebar.slider("select top results pct", 0.0, 1.0, 0.8)

te = st.sidebar.slider("select LLM temperature: ", 0.0, 1.0, 0.9)

btn = st.sidebar.button(label="Search")


if btn:
    
    # create query on image, also shows similar document in vector database (not using LLM)  -- openclip embedding function!
    embedding_function = OpenCLIPEmbeddingFunction()

    #question = "Answer with organized answers: What type of flower is in the picture? Mention some of its characteristics and how to take care of it ?"

    st.session_state["document"] = cTxts.query(
        query_embeddings=embedding_function("./" + sim.name),
        n_results=1,
    )["documents"][0][0]

    if "timgs" in st.session_state:
        st.session_state["timgs"] = []
        
    if "meta" in st.session_state:
        st.session_state["meta"] = []   
    #st.write("=>images: ", util.getMetadata(sim.name))
    
    qmdata = util.getMetadata(sim.name)
    imgs = cImgs.query(
        query_uris="./" + sim.name,
        include=["data", "metadatas"],
        n_results=6,
        where={
            "$and": [{"year": qmdata[0]}, {"month": qmdata[1]}, {"day": qmdata[2]}]
        },
    )

    for img in imgs["data"][0][1:]:
        st.session_state["timgs"].append(img)
    for mdata in imgs["metadatas"][0][1:]:
        st.session_state["meta"].append(mdata.get('location') + ": (" + mdata.get('datetime') + ")")
        

    getLLMText(question=txt, article=st.session_state['document'])


if len(st.session_state["timgs"]) > 1:
    
    st.text_area( label='LLM Description', value=st.session_state['llm_text'])
    
    st.text_area(label="Embedding Description", value=st.session_state["document"])

    dimgs = image_select(
        label="Select Image",
        images=st.session_state["timgs"],
        use_container_width=True,
        captions=st.session_state["meta"],
    )
    im = Image.fromarray(dimgs)
    nim = ImageOps.expand(im,border=(20,20,20,20), fill=(222,222,222))
    display_im = st.image(nim, use_column_width="always")
    #rot = nim.rotate(-90)
    #display_im.image(rot)
