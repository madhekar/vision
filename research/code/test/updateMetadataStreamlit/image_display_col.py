import streamlit as st
import asyncio
from math import ceil
import pandas as pd
import os
import folium as fl
from streamlit_folium import st_folium
import base64

st.set_page_config(
        page_title="zesha: Home Media Portal (HMP)",
        page_icon="/home/madhekar/work/zsource/zesha-high-resolution-logo.jpeg",
        initial_sidebar_state="auto",
        layout="wide",
    )  # (margins_css)

st.markdown('''
<style>
            .streamlit-container {
            border: 2px solid #000;
            padding: 5px;
            }
            </style>
            
            ''', unsafe_allow_html=True)

if "markers" not in st.session_state:
    st.session_state["markers"] = []

if "updated_location_list" not in st.session_state:
    st.session_state["updated_location_list"] = []

def clear_markers():
    st.session_state["markers"].clear()

def add_marker(lat, lon, label, url):
    #iconurl = fl.features.CustomIcon(url, icon_size=(50,50))
    ##
    # encoded = base64.b64encode(open(url, 'rb').read())
    # html = '<img src="data:image/jpg;base64,{}">'.format
    # ifr = fl.IFrame(html(encoded.decode('UTF-8')), width=200, height=200)
    # popup = fl.Popup(ifr, max_width= 400)
    ##
    #t = """<img src='" + url + "' width=50>"""
    #iframe = fl.IFrame(html=t, width=200, height=100)
    #pop = fl.Popup(t, max_width=2600)
    marker = fl.Marker([lat, lon], popup=url, tooltip=label)#, icon=iconurl)
    st.session_state["markers"].append(marker)

@st.cache_data
def metadata_initialize():
    df = pd.read_csv ('metadata.csv')
    #df["idx"] = range(1, len(df) +1)
    df.set_index("SourceFile", inplace=True)
    return df

if "df" not in st.session_state:
    df = metadata_initialize()
    st.session_state.df = df
else:
    df = st.session_state.df

@st.cache_data
def location_initialize():
     df_loc = pd.read_csv("locations.csv")
     df_loc.set_index("name", inplace=True)
     return df_loc

if "df_loc" not in st.session_state:
     df_loc = location_initialize()
     st.session_state.df_loc = df_loc
else:
     df_loc = st.session_state.df_loc

st.markdown("<p class='big-font-title'>Home Media Portal</p>", unsafe_allow_html=True)
st.logo("/home/madhekar/work/home-media-app/app/zesha-high-resolution-logo.jpeg")

# extract files
files = pd.read_csv("metadata.csv")["SourceFile"]

def update_all_latlon():
    if len(st.session_state.df_loc) > 0 :
      for loc in st.session_state["updated_location_list"]:
        print('-->', loc)
        st.session_state.df.at[loc[0],"GPSLatitude"] = st.session_state.df_loc.at[loc[2],'lat']   
        st.session_state.df.at[loc[0], "GPSLongitude"] = st.session_state.df_loc.at[loc[2], "lon"]
      st.session_state["updated_location_list"].clear()  

def save_metadata():
   st.session_state.df.to_csv("metadata.csv", sep=",")   
   st.session_state.df_loc.to_csv("locations.csv", sep=",")

async def main():

    layout = st.columns([.12,.88])
    
    with layout[0]:
        
        st.divider()
        st.markdown("Select Display")
        batch_size = st.select_slider("Batch size:", range(10, 700, 10))
        row_size = st.select_slider("Row size:", range(1, 10), value=7)
        num_batches = ceil(len(files) / batch_size)
        page = st.selectbox("Page#:", range(1, num_batches + 1))

        st.divider()
        st.markdown("Locations")
        #st.button(label="Add/Update", on_click=add_location())
        st.session_state.df_loc = st.data_editor(st.session_state.df_loc, num_rows="dynamic", use_container_width=True, disabled=["widgets"])

        st.divider()
        st.markdown("Metadata")
        st.button(label="Save", on_click=save_metadata())

    with layout[1]:
        m = fl.Map(location=[32.968700, -117.184200], zoom_start=5)

        fg = fl.FeatureGroup(name="zesha")

        for marker in st.session_state["markers"]:
            fg.add_child(marker)

        # m.add_child(fl.LatLngPopup())

        map = st_folium(m, width="100%", feature_group_to_add=fg)

        data = None
        if map.get("last_clicked"):
            data = (map["last_clicked"]["lat"], map["last_clicked"]["lng"])

        if data is not None:
            # st.write(data)
            print(data)

    st.divider()
    batch = files[(page - 1) * batch_size : page * batch_size]
    grid = st.columns(row_size)
    col = 0

    for image in batch:
        with grid[col]:
            # st.write(page -1 ,batch_size,col, image)

            c1, c2, c3 = st.columns([1, 1, 1])
            lat = st.session_state.df.at[image, "GPSLatitude"]
            lon = st.session_state.df.at[image, "GPSLongitude"]
            label = os.path.basename(image)
            if lat != "-":
                c1.text_input(
                    value=lat, label=f"Lat_{image}", label_visibility="hidden"
                )  # ,on_change=update_latitude(col, image))
                c2.text_input(
                    value=lon, label=f"Lon_{image}", label_visibility="hidden"
                )  # , on_change=update_longitude(col, image))
                c3.text_input(
                    value=st.session_state.df.at[image, "DateTimeOriginal"],
                    label=f"dt_{image}",
                    label_visibility="hidden",
                )  # , on_change=update_date(col, image), args=(image, 'label'))
            else:
                r = c1.selectbox(label=f"location_{image}", 
                                 label_visibility="hidden", 
                                 options=st.session_state.df_loc.index.values, 
                                 index=None, on_change=update_all_latlon())
                if r:
                  st.session_state["updated_location_list"].append((image, col, r))
                c3.text_input(
                    value=st.session_state.df.at[image, "DateTimeOriginal"],
                    label=f"dt_{image}",
                    label_visibility="hidden",
                )  # , on_change=update_date(col, image), args=(image, 'label'))
            st.image(image, caption=label)

            if lat != "-":
                add_marker(lat, lon, label, image)

        col = (col + 1) % row_size

if __name__ == "__main__":
    asyncio.run(main())