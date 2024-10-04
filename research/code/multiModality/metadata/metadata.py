import streamlit as st
import asyncio
import yaml
from math import ceil
import pandas as pd
import os
import folium as fl
from streamlit_folium import st_folium
import streamlit_init as sti
import util 


# initialize streamlit container UI settings
sti.initUI()

smp, smf, mmp, mmf = util.config_load()

st.markdown("<p class='big-font-title'>Metadata Editor - Home Media Portal</p>", unsafe_allow_html=True)
st.logo("/home/madhekar/work/home-media-app/app/zesha-high-resolution-logo.jpeg")

if "markers" not in st.session_state:
    st.session_state["markers"] = []

if "updated_location_list" not in st.session_state:
    st.session_state["updated_location_list"] = []

if "updated_datetime_list" not in st.session_state:
    st.session_state["updated_datetime_list"] = []    

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
    df = pd.read_csv (os.path.join(mmp, mmf)) #('metadata.csv')
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
     df_loc = pd.read_csv (os.path.join(smp, smf)) #("locations.csv")
     df_loc.set_index("name", inplace=True)
     return df_loc

if "df_loc" not in st.session_state:
     df_loc = location_initialize()
     st.session_state.df_loc = df_loc
else:
     df_loc = st.session_state.df_loc

# extract files
files = pd.read_csv(os.path.join(mmp, mmf))["SourceFile"]

def update_all_latlon():
    if len(st.session_state.updated_location_list) > 0 :
        print(st.session_state["updated_location_list"])
        for loc in st.session_state["updated_location_list"]:
            print("-->", loc)
            lat = st.session_state.df_loc.at[loc[2], "lat"]
            lon = st.session_state.df_loc.at[loc[2], "lon"]
            st.session_state.df.at[loc[0], "GPSLatitude"] = lat
            st.session_state.df.at[loc[0], "GPSLongitude"] = lon
            util.setGpsInfo(loc[0], lat=lat, lon=lon)
        st.session_state["updated_location_list"].clear()  

def update_all_datetime_changes(image, col):
    #print(st.session_state[f'{col}_{image}'])
    dt = st.session_state[f"{col}_{image}"]
    st.session_state.df.at[image, "DateTimeOriginal"] = dt
    util.setDateTimeOriginal(image, dt)


def save_metadata():
    st.session_state.df.to_csv(os.path.join(mmp, mmf), sep=",")
    st.session_state.df_loc.to_csv(os.path.join(smp, smf), sep=",")


async def main():
    l1,l2 = st.columns([.12,.88])
    with l1:
        l1.divider()
        l1.markdown("Display Images")
        batch_size = l1.select_slider("Batch size:", range(10, 700, 10))
        row_size = l1.select_slider("Row size:", range(1, 10), value=7)
        num_batches = ceil(len(files) / batch_size)
        page = l1.selectbox("Page#:", range(1, num_batches + 1))

        l1.divider()
        l1.markdown("Locations")
        #st.button(label="Add/Update", on_click=add_location())
        st.session_state.df_loc = st.data_editor(st.session_state.df_loc, num_rows="dynamic", use_container_width=True, height=200)
        l1.button(label="Save Metadata", on_click=save_metadata(), use_container_width=True)

    with l2:
        m = fl.Map(location=[32.968700, -117.184200], zoom_start=7)

        fg = fl.FeatureGroup(name="zesha")

        for marker in st.session_state["markers"]:
            fg.add_child(marker)

        m.add_child(fl.LatLngPopup())

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
    st.cache_resource
    for image in batch:
        with grid[col]:
            c1, c2, c3 = st.columns([1, 1, 1])
            lat = st.session_state.df.at[image, "GPSLatitude"]
            lon = st.session_state.df.at[image, "GPSLongitude"]
            dt = st.session_state.df.at[image, "DateTimeOriginal"]
            label = os.path.basename(image)
            if lat != "-":
                c1.text_input(value=lat, label=f"Lat_{image}", label_visibility="hidden")  # ,on_change=update_latitude(col, image))
                c2.text_input(value=lon, label=f"Lon_{image}", label_visibility="hidden")  # , on_change=update_longitude(col, image))
                c3.text_input(value=dt,label=f"dt_{image}", label_visibility="hidden", on_change=update_all_datetime_changes, key=f"dt_{image}", args=(image, 'dt'))
            else:
                r = c2.selectbox(label=f"location_{image}", label_visibility="hidden",  options=st.session_state.df_loc.index.values, index=None, on_change=update_all_latlon())
                if r:
                  st.session_state["updated_location_list"].append((image, col, r))
                c3.text_input(value=dt,label=f"dt_{image}", label_visibility="hidden", on_change=update_all_datetime_changes, key=f"dt_{image}", args=(image, 'dt')) 
            st.image(image, caption=label, output_format="JPG")
            if lat != "-":
                add_marker(lat, lon, label, image)

        col = (col + 1) % row_size

if __name__ == "__main__":
        asyncio.run(main())