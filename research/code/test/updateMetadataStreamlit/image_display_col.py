import streamlit as st
from math import ceil
import pandas as pd
import os
import folium as fl
from streamlit_folium import st_folium

st.set_page_config(
        page_title="zesha: Home Media Portal (HMP)",
        page_icon="/home/madhekar/work/zsource/zesha-high-resolution-logo.jpeg",
        initial_sidebar_state="auto",
        layout="wide",
    )  # (margins_css)

if "markers" not in st.session_state:
    st.session_state["markers"] = []

def marker(lat, lon):
    return fl.Marker(location=[lat, lon])

def get_pos(lat, lng):
    return lat, lng

m = fl.Map(location=[39.8283, -98.5795], zoom_start=5)

fg = fl.FeatureGroup(name="zesha")

m.add_child(fl.LatLngPopup())

map = st_folium(m, width='100%', feature_group_to_add= fg)

data = None
if map.get("last_clicked"):
    data = get_pos(map["last_clicked"]["lat"], map["last_clicked"]["lng"])

if data is not None:
    #st.write(data)
    print(data)  


dft = pd.read_csv(os.path.join(os.path.abspath(".."), "out.csv"))
files = dft["SourceFile"]

@st.cache_data
def initialize():
    df = pd.read_csv (os.path.join(os.path.abspath('..') , 'out.csv'))
    #df["idx"] = range(1, len(df) +1)
    df.set_index("SourceFile", inplace=True)
    return df


if "df" not in st.session_state:
    df = initialize()
    st.session_state.df = df
else:
    df = st.session_state.df

#st.write(st.session_state.df.head())
#files = st.session_state.df["SourceFile"]

controls = st.columns(3)
with controls[0]:
    batch_size = st.select_slider("Batch size:", range(10, 110, 10))
with controls[1]:
    row_size = st.select_slider("Row size:", range(1, 6), value=5)
num_batches = ceil(len(files) / batch_size)
with controls[2]:
    page = st.selectbox("Page", range(1, num_batches + 1))


def update(image, col):
    df.at[image, col] = st.session_state[f"{col}_{image}"]
    if st.session_state[f"incorrect_{image}"] == False:
        st.session_state[f"label_{image}"] = ""
        df.at[image, "label"] = ""

def update_latitude(col, image):
  df.at[image, col] = st.session_state[f"{col}_{image}"]

def update_longitude(col, image):
    df.at[image, col] = st.session_state[f"{col}_{image}"]


def update_date(col, image):
    df.at[image, col] = st.session_state[f"{col}_{image}"]

batch = files[(page - 1) * batch_size : page * batch_size]
grid = st.columns(row_size)
#st.write(batch)
col = 0

for image in batch:
    with grid[col]:
        st.write(page -1 ,batch_size,col, image)

        c1,c2,c3 =st.columns([1,1,1])
        lat = df.at[image, "GPSLatitude"] 
        lon = df.at[image, "GPSLongitude"]
        c1.text_input(value=lat, label=f"Lat_{image}", label_visibility="hidden") #,on_change=update_latitude(col, image))
        c2.text_input(value=lon,label=f"Lon_{image}",label_visibility="hidden",)  # , on_change=update_longitude(col, image))
        c3.text_input(
            value=df.at[image, "DateTimeOriginal"],
            label=f"dt_{image}",
            label_visibility="hidden",
        )  # , on_change=update_date(col, image), args=(image, 'label'))
        st.image(image, caption=os.path.basename(image))
        if lat != "-":
           fl.Marker(location=[df.at[image, "GPSLatitude"], df.at[image, "GPSLongitude"]]).add_to(fg)
        # st.checkbox(
        #     "Incorrect",
        #     key=f"incorrect_{image}",
        #     value=df.at[image, "incorrect"],
        #     on_change=update,
        #     args=(image, "incorrect"),
        # )
        # if df.at[image, "incorrect"]:
        #     st.text_input(
        #         "New label:",
        #         key=f"label_{image}",
        #         value=df.at[image, "label"],
        #         on_change=update,
        #         args=(image, "label"),
        #     )
        # else:
        #     st.write("##")
        #     st.write("##")
        #     st.write("###")
    col = (col + 1) % row_size

# m = fl.Map(location=[39.8283, -98.5795], zoom_start=5)

# fg = fl.FeatureGroup(name="zesha")

# m.add_child(fl.LatLngPopup())

# map = st_folium(m, width="100%", feature_group_to_add=fg)

# data = None
# if map.get("last_clicked"):
#     data = get_pos(map["last_clicked"]["lat"], map["last_clicked"]["lng"])

# if data is not None:
#     # st.write(data)
#     print(data)

# st.write("## Corrections")
# df[df["incorrect"] == True]
