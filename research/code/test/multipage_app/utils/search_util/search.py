import streamlit as st
import datetime
import pandas as pd
import ast

# from streamlit_option_menu import option_menu
from streamlit_image_select import image_select
from utils.config_util import config
from PIL import Image, ImageOps
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
import datetime

import chromadb as cdb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from chromadb.config import Settings

MIN_DT = datetime.datetime(1998, 1, 1)
MAX_DT = datetime.datetime.now()


@st.cache_resource(show_spinner=True)
def init_vdb(vdp, icn, tcn):
    # vector database persistance
    client = cdb.PersistentClient( path=vdp, settings=Settings(allow_reset=True))
    
    # openclip embedding function!
    embedding_function = OpenCLIPEmbeddingFunction()

    # Image collection inside vector database 'chromadb'
    image_loader = ImageLoader()

    # collection images defined
    collection_images = client.get_or_create_collection(
      name=icn, 
      embedding_function=embedding_function, 
      data_loader=image_loader
      )
    
    #Text collection inside vector database 'chromadb'
    collection_text = client.get_or_create_collection(
      name=tcn,
      embedding_function=embedding_function,
    )

    return client, collection_images, collection_text

def updateMetadata(client, image_collection,  id, desc, names, dt, loc):
    # vector database persistance
    #client = cdb.PersistentClient(path=storage_path, settings=Settings(allow_reset=True))
    col = client.get_collection(image_collection)
    col.update(
        ids=id,
        metadatas={"description": desc, "names": names, "datetime" : dt, "location": loc}
    )

def search_fn(client, cImgs, cTxts):
    # create default application Tabs
    image, video, text = st.tabs(["Image", "Video", "Text"])

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
                help="select image to search similar images",
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

    # seach button pressed
    if search_btn:
        # create query on image, also shows similar document in vector database (not using LLM)  -- openclip embedding function!
        embedding_function = OpenCLIPEmbeddingFunction()

        # reset session_state for temperory images for view
        if "timgs" in st.session_state:
            st.session_state["timgs"] = []

        # reset session_state for metadata
        if "meta" in st.session_state:
            st.session_state["meta"] = []

        if modality_selected == "image":
            # execute text collection query --- TBD fix

            # st.session_state["document"] = cTxts.query(
            #     query_embeddings=embedding_function("./" + similar_image.name),
            #     n_results=5,
            # )["documents"][0][0]

            # get location and datetime metadata for an image
            # qmdata = util.getMetadata(sim.name)
            # dt_format = "%Y-%m-%d %H:%M:%S"
            # st.write(qmdata[3])
            # d = parser.parse(qmdata[3]).timestamp()
            # st.write(
            #    d,
            #    st.session_state["dt_range"][0].timestamp(),
            #    st.session_state["dt_range"][1].timestamp(),
            # )

            # execute image query with search criteria
            st.session_state["imgs"] = cImgs.query(
                query_uris="./" + similar_image.name,
                include=["data", "metadatas"],
                n_results=8,
            )

            st.write(st.session_state["imgs"])

        elif modality_selected == "text":
            # execute text collection query --- TBD fix
            # st.session_state["document"] = cTxts.query(
            #     query_texts=modalityTxt,
            #     n_results=5,
            # )["documents"][0][0]

            # execute image query with search criteria
            st.session_state["imgs"] = cImgs.query(
                query_texts=modalityTxt, include=["data", "metadatas"], n_results=10
            )

        for img in st.session_state["imgs"]["data"][0][1:]:
            st.session_state["timgs"].append(img)
        for mdata in st.session_state["imgs"]["metadatas"][0][1:]:
            st.write(mdata)
            st.session_state["meta"].append(
                "Desc:["
                + mdata.get("text")
                + "] ) People: ["
                + mdata.get("names")
                + "] Location: ["
                + mdata.get("loc")
                + "] Date: ["
                + str(datetime.datetime.fromtimestamp(float(mdata.get("ts"))))
                + "]"
            )
    # Image TAB
    with image:
        if st.session_state["timgs"] and len(st.session_state["timgs"]) > 1:
            index = image_select(
                label="Similar Images",
                images=st.session_state["timgs"],
                use_container_width=True,
                # captions=st.session_state["meta"],
                index=0,
                return_value="index",
            )
            # img, map = st.tabs(["Img", "Map"])
            c1, c2 = st.columns([9, 1])

            # c2.divider()
            col21, col22, col23 = c2.columns([1, 1, 1], gap="small")
            with col21:
                right = st.button(label="## &#x21B7;")
            with col22:
                left = st.button(label="## &#x21B6;")
            with col23:
                flip = st.button(label="## &#x21C5;")

            # with img:
            im = Image.fromarray(st.session_state["timgs"][index])
            nim = ImageOps.expand(im, border=(2, 2, 2, 2), fill=(200, 200, 200))
            imageLoc = c1.empty()
            display_im = imageLoc.image(nim, use_column_width="always")
            # st.button(st.image(nim, use_column_width="always"))

            if right:
                nim = nim.rotate(-90)
                imageLoc.image(nim, use_column_width="always")
            if left:
                nim = nim.rotate(90)
                imageLoc.image(nim, use_column_width="always")
            if flip:
                nim = nim.rotate(180)
                imageLoc.image(nim, use_column_width="always")
            # with map:
            # st.write(
            #     "<p class='big-font'>sorry, no map is implemented found in search criteria!</p>",
            #     unsafe_allow_html=True,
            # )

            # c2.divider()
            colt, cole = c2.columns([0.7, 0.3])
            with colt:
                st.markdown(
                    "<p class='big-font-subh'>Gleeful Desc</p>", unsafe_allow_html=True
                )
            with cole:
                edit = st.button(label="## &#x270D;")

            if edit:
                updateMetadata(
                    client,
                    cImgs,
                    id=st.session_state["imgs"]["ids"][0][index],
                    desc=st.session_state["imgs"]["metadatas"][0][1:][index]["text"],
                    names=st.session_state["imgs"]["metadatas"][0][1:][index]["names"],
                    dt=st.session_state["imgs"]["metadatas"][0][1:][index]["ts"],
                    loc=st.session_state["imgs"]["metadatas"][0][1:][index]["loc"],
                )
            o_desc = f'<p class="big-font">{st.session_state["imgs"]["metadatas"][0][1:][index]["text"]}</p>'
            c2.markdown(o_desc, unsafe_allow_html=True)

            c2.write("<p class='big-font-subh'>People</p>", unsafe_allow_html=True)
            o_names = f'<p class="big-font">{st.session_state["imgs"]["metadatas"][0][1:][index]["attrib"]} - {st.session_state["imgs"]["metadatas"][0][1:][index]["names"]}</p>'
            c2.markdown(o_names, unsafe_allow_html=True)

            c2.write("<p class='big-font-subh'>Date Time</p>", unsafe_allow_html=True)
            o_datetime = f'<p class="big-font">{str(datetime.datetime.fromtimestamp(float(st.session_state["imgs"]["metadatas"][0][1:][index]["ts"])))}</p>'
            c2.markdown(o_datetime, unsafe_allow_html=True)

            c2.write("<p class='big-font-subh'>Location</p>", unsafe_allow_html=True)
            o_location = f'<p class="big-font">{st.session_state["imgs"]["metadatas"][0][1:][index]["loc"]}</p>'
            c2.markdown(o_location, unsafe_allow_html=True)

            ll = ast.literal_eval(st.session_state["imgs"]["metadatas"][0][1:][index]["latlon"])     
            lat = ll[0] #float(st.session_state["imgs"]["metadatas"][0][1:][index]["latlon"][0])
            lon = ll[1] # float(st.session_state["imgs"]["metadatas"][0][1:][index]["latlon"][1])

            map_data = pd.DataFrame({"lat": [lat], "lon": [lon]})
            c2.markdown("<p class='big-font-subh'>Map</p>", unsafe_allow_html=True)
            c2.map(map_data, zoom=12, size=100, color="#ff00ff")
        else:
            st.write(
                "<p class='big-font'>sorry, no similar images found in search criteria!</p>",
                unsafe_allow_html=True,
            )

    #  Video TAB
    with video:
        # st.header("Similar Videos")
        st.write(
            "<p class='big-font'>sorry, no similar videos found in search criteria!</p>",
            unsafe_allow_html=True,
        )

    #  Documents Tab
    with text:
        if st.session_state["document"] and len(st.session_state["document"]) > 1:
            st.text_area(label="Related text", value=st.session_state["document"])
        else:
            st.write(
                "<p class='big-font'>sorry, no similar documents found in search criteria!</p>",
                unsafe_allow_html=True,
            )

def execute():

    vdb, icn, tcn, vcn, acn = config.search_config_load()

    client, img_collection, txt_collection  = init_vdb(vdb, icn, tcn)

    search_fn(client, img_collection, txt_collection)

    