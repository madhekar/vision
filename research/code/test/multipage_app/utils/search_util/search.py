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

# # initialize streamlit container UI settings
# streamlit_init.initUI()

# # load data
# cImgs, cTxts = init()

# # st.markdown("<p class='big-font-title'>Home Media Portal</p>", unsafe_allow_html=True)
# st.logo("/home/madhekar/work/home-media-app/app/zesha-high-resolution-logo.jpeg")
def search_fn():

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
        # st.markdown(
        #     "<p class='big-font-header'>@Home Media Portal</p>", unsafe_allow_html=True
        # )

        # st.divider()

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
    # sm.add_messages("metadata", "s| starting to analyze missing metadata files...")

    # imp, mmp, mmf = config.missing_metadata_config_load()

    # input_image_path = os.path.join(imp, source_name)
    # try:
    #     args = shlex.split(
    #         f"exiftool -gps:GPSLongitude -gps:GPSLatitude -DateTimeOriginal -csv -T -r -n {input_image_path}"
    #     )
    #     proc = subprocess.run(args, capture_output=True)
    # except Exception as e:
    #     print(f"error {e}")

    # # print(proc.stderr)
    # # print(proc.stdout)

    # # arc_folder_name_dt = mu.get_foldername_by_datetime()

    # output_file_path = os.path.join(mmp, source_name)  # , arc_folder_name_dt)

    # if not os.path.exists(output_file_path):
    #     os.makedirs(output_file_path)

    # with open(os.path.join(output_file_path, mmf), "wb") as output:
    #     output.write(proc.stdout)

    # create_missing_report(os.path.join(output_file_path, mmf))

    # sm.add_messages(
    #     "metadata",
    #     f"w| finized to analyze missing metadata files created {output_file_path}.",
    # )
    