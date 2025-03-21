import streamlit as st
import datetime
import pandas as pd

# from streamlit_option_menu import option_menu
from streamlit_image_select import image_select
from PIL import Image, ImageOps
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
import datetime

MIN_DT = datetime.datetime(1998, 1, 1)
MAX_DT = datetime.datetime.now()


def config_load():
    with open("app_conf.yaml") as prop:
        dict = yaml.safe_load(prop)

        print(
            "* * * * * * * * * * * Metadata Generator Properties * * * * * * * * * * * *"
        )
        for k in dict.keys():
            print(f"{k} :  {dict[k]}  \n")

        print(
            "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *"
        )
        vectordb_dir_path = dict["vectordb"]["vectordb_path"]
        image_collection_name = dict["vectordb"]["image_collection_name"]
        text_collection_name = dict["vectordb"]["text_collection_name"]

    return (vectordb_dir_path, image_collection_name, text_collection_name)


def search_fn():
    # create default application Tabs
    # image, video, text = st.tabs(["Image", "Video", "Text"])

    # init session variables
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
        st.session_state["dt_range"] = (
            datetime.datetime(2010, 1, 1),
            datetime.datetime(2019, 1, 1),
        )

    # define application sidebar
    with st.sidebar:

        modality_selected = st.selectbox(
            label="## Search Modality",
            options=("text", "image"),
            index=1,
            help="select search modality type",
        )

        st.divider()

        multi_modality_select = st.multiselect(
            label="## Show Modalities",
            options=["image", "text", "video", "audio"],
            default=["image", "text"],
            help="select one or more search result modalities",
        )

        st.divider()

        if modality_selected == "image":
            similar_image = st.file_uploader(
                label="## Select Image",
                label_visibility="hidden",
                type=["png", "jpeg", "mpg", "jpg", "PNG", "JPG"],
                help="select example image to search similar images",
            )
            im = st.empty()
            if similar_image:
                im = Image.open(similar_image)
                name = similar_image.name
                st.sidebar.image(im, caption="")
                # st.sidebar.write(st.session_state["llm_text"])
                with open(name, "wb") as f:
                    f.write(similar_image.getbuffer())
        elif modality_selected == "text":
            modalityTxt = st.text_input(
                label="## Search text",
                placeholder="search modality types for...",
                disabled=False,
            )

        st.divider()

        def date_change():
            st.session_state["dt_range"] = st.session_state.mySlider

        date_range = st.slider(
            label="## Date range",
            key="mySlider",
            value=st.session_state["dt_range"],
            min_value=MIN_DT,
            max_value=MAX_DT,
            step=datetime.timedelta(days=1),
            on_change=date_change,
            help="search result date range",
        )

        st.divider()

        search_btn = st.button(label="## Search")


def execute():
    search_fn()

    