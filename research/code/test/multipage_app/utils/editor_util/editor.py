import streamlit as st
from math import ceil
import pandas as pd
import os
import folium as fl
from streamlit_folium import st_folium
from utils.util import location_util as lu 
from utils.config_util import config
from utils.sqlite_util import location
from PIL import Image
from utils.util import fast_parquet_util as fpu


@st.cache_resource
def metadata_initialize(mmp,mmf):
    df = pd.read_csv (os.path.join(mmp, mmf)) #('metadata.csv')
    df.set_index("SourceFile", inplace=True)
    print(df.head(10))
    return df

@st.cache_resource
def location_initialize(smp, smf):
    df = None
    try:
        df = fpu.read_parquet_file(os.path.join(smp, smf))
        print(df)
    except Exception as e:
        print(
            f"exception occured in loading location metadata: {smf} with exception: {e}"
        )  
    return df    
    # db_con = location.Location(dbpath=sdp, dbname=sdn)
    # db_con.create_location_tbl_if_not_exists()
    # n = db_con.get_number_of_rows()
    # print(n)
    # if n[0][0] != 0:
    #     t_arr = db_con.read_location()
    #     df_loc = pd.DataFrame(t_arr)
    #     df_loc.columns = ["name", "desc", "lat", "lon"]
    #     #df_loc.set_index('name', inplace=True)
    # else:
    #     df_loc = pd.DataFrame(columns=["name", "desc", "lat", "lon"])
    #     #df_loc.set_index('name', inplace=True)
    # return df_loc

@st.cache_resource
def initialize():
    """
    metadata:
      static_metadata_path: /home/madhekar/work/home-media-app/data/app-data/static-metadata/
      static_metadata_file: static_locations.parquet
      missing_metadata_path: /home/madhekar/work/home-media-app/data/input-data/error/img/missing-data/
      missing_metadata_file: missing-metadata-wip.csv
      home_latitude: 32.968700
      home_longitude: -117.184200
    """
    
    smp, smf, mmp, mmf, hlat, hlon = config.editor_config_load()

    reload_bug = True

    if "markers" not in st.session_state:
        st.session_state["markers"] = []

    if "updated_location_list" not in st.session_state:
        st.session_state["updated_location_list"] = []

    if "updated_datetime_list" not in st.session_state:
        st.session_state["updated_datetime_list"] = []   

    if "editor_audit_msg" not in st.session_state:
        st.session_state["editor_audit_msg"] = []   
        
    if "df" not in st.session_state:
        df = metadata_initialize(mmp, mmf)
        st.session_state.df = df
    else:
        df = st.session_state.df

    if "df_loc" not in st.session_state:
        df_loc = location_initialize(smp, smf)
        st.session_state.df_loc = df_loc
    else:
        df_loc = st.session_state.df_loc     

    return smp, smf, mmp, mmf, hlat, hlon, reload_bug

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



# extract files
#files = pd.read_csv(os.path.join(mmp, mmf))["SourceFile"]

def update_all_latlon():
    if len(st.session_state.updated_location_list) > 0 :
        print(st.session_state["updated_location_list"])
        for loc in st.session_state["updated_location_list"]:
            lat = st.session_state.df_loc.at[loc[2], "lat"]
            lon = st.session_state.df_loc.at[loc[2], "lon"]
            st.session_state.df.at[loc[0], "GPSLatitude"] = lat
            st.session_state.df.at[loc[0], "GPSLongitude"] = lon
            lu.setGpsInfo(loc[0], lat=lat, lon=lon)
        st.session_state["updated_location_list"].clear()  

def update_all_datetime_changes(image, col):
    dt = st.session_state[f"{col}_{image}"]
    st.session_state.df.at[image, "DateTimeOriginal"] = dt
    lu.setDateTimeOriginal(image, dt)

def select_location_by_country_and_state(parquet_file_path):

    rdf = fpu.read_parquet_file(file_path=parquet_file_path)

    c_country, c_state, c_location = st.columns([.2,.2,1])
    
    with c_country:
      selected_country = st.selectbox('select country', rdf['country'].unique())

    with c_state:
      frdf = rdf[rdf["country"] == selected_country]
      s_frdf = frdf.sort_values(by='state')
      selected_state = st.selectbox('select state', s_frdf['state'].unique())

    with c_location:
       ffrdf = frdf[frdf['state'] == selected_state]
       s_ffrdf = ffrdf.sort_values(by='name')
       selected_location = st.selectbox('select location name/ description', s_ffrdf['name'].unique())

       st.write(s_ffrdf[s_ffrdf['name'] == selected_location].iloc[0])

def save_metadata(sdp, sdn, mmp, mmf):
    st.session_state.df.to_csv(os.path.join(mmp, mmf), sep=",")
    # persist_static_locations(sdp, sdn)
    print(st.session_state.df)

def execute():

    smp, smf, mmp, mmf, hlat, hlon, reload_bug = initialize()
    
    # extract files
    files = pd.read_csv(os.path.join(mmp, mmf))["SourceFile"]

    st.sidebar.header("Display Criteria",divider="gray")
    cb,cr,cp = st.sidebar.columns([1,1,1])
    with cb:
        batch_size = st.select_slider("Batch Size:", range(10, 700, 10))
    with cr:   
        row_size = st.select_slider("Row Size:", range(1, 10), value=7)   
        num_batches = ceil(len(files) / batch_size)
    with cp:   
        page = st.selectbox("Page Number:", range(1, num_batches + 1))


    st.sidebar.header('Locations', divider="gray")
    config = {
        'name' : st.column_config.TextColumn('Name', width='small', required=True),
        'desc' : st.column_config.TextColumn('Description', width='small', required=True),
        'lat' : st.column_config.NumberColumn('Latitude', min_value=-90.0, max_value=90.0, required=True),
        'lon' : st.column_config.NumberColumn('Logitude',min_value=-180.0, max_value= 180.0, required=True)
    }
    st.session_state.df_loc = st.sidebar.data_editor(st.session_state.df_loc, column_config=config, num_rows="dynamic", use_container_width=True, height=350, hide_index=True) #

    save_btn = st.sidebar.button(label="Save Metadata",  use_container_width=True) #on_click=save_metadata(sdp, sdn, mmp, mmf)
    if save_btn:
        save_metadata(smp, smf, mmp, mmf)

    m = fl.Map(location=[hlat, hlon], zoom_start=4, min_zoom=3, max_zoom=10)

    fg = fl.FeatureGroup(name="zesha")

    for marker in st.session_state["markers"]:
        fg.add_child(marker)

    m.add_child(fl.LatLngPopup())

    map = st_folium(m, width="100%", feature_group_to_add=fg)

    data = None
    if map.get("last_clicked"):
        data = (map["last_clicked"]["lat"], map["last_clicked"]["lng"])

    if data is not None:
        st.session_state.editor_audit_msg.append(data)
    #...
    st.subheader("IMAGES", divider='gray')    

    batch = files[(page - 1) * batch_size : page * batch_size]
    grid = st.columns(row_size, gap="small", vertical_alignment="top")
    col = 0

    for image in batch:
        with grid[col]:
            c1, c2 = st.columns([1.0, 1.0], gap="small", vertical_alignment="top")
            lat = st.session_state.df.at[image, "GPSLatitude"]
            lon = st.session_state.df.at[image, "GPSLongitude"]
            dt = st.session_state.df.at[image, "DateTimeOriginal"]
            label = os.path.basename(image)
            if lat != "-":
                c2.empty()
                c2.text_input(value=lat, label=f"Lat_{image}", label_visibility="collapsed")  
                c2.empty()
                c2.text_input(value=lon, label=f"Lon_{image}", label_visibility="collapsed") 
                c2.empty()
                c2.text_input(value=dt,label=f"dt_{image}", label_visibility="collapsed", on_change=update_all_datetime_changes, key=f"dt_{image}", args=(image, 'dt'))
            else:
                r = c2.selectbox(label=f"location_{image}", label_visibility="collapsed",  options=st.session_state.df_loc.name.values, index=None, on_change=update_all_latlon())
                if r:
                  st.session_state["updated_location_list"].append((image, col, r))
                c2.text_input(value=dt,label=f"dt_{image}", label_visibility="collapsed", on_change=update_all_datetime_changes, key=f"dt_{image}", args=(image, 'dt')) 
            image = Image.open(image)  
            image.thumbnail((200,200), Image.Resampling.LANCZOS)
            c1.image(image, caption=label, output_format="JPG")
            if lat != "-":
                add_marker(lat, lon, label, image)
            st.divider()    

        col = (col + 1) % row_size
    # if reload_bug:
    #     reload_bug = False
    #     st.rerun()

if __name__ == "__main__":
    execute()