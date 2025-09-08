import streamlit as st
from math import ceil
import pandas as pd
import os
import folium as fl
from streamlit_folium import st_folium
from streamlit_folium import folium_static
from folium.plugins import FastMarkerCluster
from utils.config_util import config
from utils.util import location_util as lu 
from utils.util import storage_stat as ss

from PIL import Image
from utils.util import fast_parquet_util as fpu

user_device = st.empty()
def get_env():
    (rdp, smp, smf, mmp, mmf, mmff, mmef, hlat, hlon, bsmx,rszp) = config.editor_config_load()
    return (rdp, smp, smf, mmp, mmf, mmff, mmef, hlat, hlon, bsmx, rszp)

@st.cache_resource
def metadata_initialize(mmp,us,mmf):
       df = pd.read_csv (os.path.join(mmp, us, mmf)) #('metadata.csv')
       df.set_index("SourceFile", inplace=True)
       return df

@st.cache_resource
def location_initialize(smp,user_source, smf):
    try:
        df = fpu.read_parquet_file(os.path.join(smp, user_source, smf))
        print('====',df.head())
    except Exception as e:
        st.error(f"exception occured in loading location metadata: {smf} with exception: {e}")  
    return df    

def remove_initialization_on_device_change():
    print('----> enter remove state function')
    if 'makers' in st.session_state:
        del st.session_state['makers']
    if "updated_datetime_list" in st.session_state:
        del st.session_state["updated_datetime_list"]
    if "editor_audit_msg" in st.session_state:
        del st.session_state["editor_audit_msg"]
    if "df" in st.session_state:
        del st.session_state["df"]
    if "df_loc" in st.session_state:
        del st.session_state["df_loc"]
    if "edited_image_attributes" in st.session_state:
        del st.session_state["edited_image_attributes"]


def initialize(smp, smf, mmp, mmf, mmef, hlat, hlon, user_source):

    try:

        if "markers" not in st.session_state:
            st.session_state["markers"] = []

        # if "updated_location_list" not in st.session_state:
        #     st.session_state["updated_location_list"] = []

        if "updated_datetime_list" not in st.session_state:
            st.session_state["updated_datetime_list"] = []   

        if "editor_audit_msg" not in st.session_state:
            st.session_state["editor_audit_msg"] = []   
            
        if "df" not in st.session_state:
            df = metadata_initialize(mmp, user_source, mmf)
            st.session_state.df = df
        else:
            df = st.session_state.df

        if "df_loc" not in st.session_state:
            df = location_initialize(smp, user_source, smf)
            st.session_state.df_loc = df
        else:
            df_loc = st.session_state.df_loc   

        if "edited_image_attributes" not in st.session_state:
            st.session_state["edited_image_attributes"] = pd.DataFrame(columns=('SourceFile', 'GPSLatitude', 'GPSLongitude', 'DateTimeOriginal'))     

    except Exception as e:      
        st.error(f"Exception occurred in initializing Medata Editor: {e}")

def clear_markers():
    st.session_state["markers"].clear()

def add_marker(lat, lon, label, url):
    print(f'added marker at: {lat} : {lon}')
    marker = fl.Marker([lat, lon], popup=url, tooltip=label)
    st.session_state["markers"].append(marker)

@st.fragment 
def update_latitude_longitude(image, latitude, longitude, name):
    #print(st.session_state.df.head())
    st.session_state.df.at[image, "GPSLatitude"] =latitude
    st.session_state.df.at[image, "GPSLongitude"] = longitude
    lu.setGpsInfo(image, latitude, longitude)
    lu.setImageDescription(image, name)

@st.fragment
def update_all_datetime_changes(image, col):
    #print(st.session_state.df.head())
    dt = st.session_state[f"{col}_{image}"]
    st.session_state.df.at[image, "DateTimeOriginal"] = dt
    lu.setDateTimeOriginal(image, dt)

def select_location_by_country_and_state(rdf):
    
    c_location_type, c_country, c_state = st.sidebar.columns([.1,.1,.1], gap="small")
    
    with c_location_type:
        is_public_location = st.selectbox('type', options=('personal','public','both'), placeholder="select type of locations to display...")
        if is_public_location == 'personal':
            rdf = rdf[rdf['name'].str.len() >  15]
        elif is_public_location == 'public':
            rdf = rdf[rdf["name"].str.len() <= 15]   
        else:
            pass 

    with c_country:
      selected_country = st.selectbox('country', rdf['country'].unique())

    with c_state:
        frdf = rdf[rdf["country"] == selected_country]
        s_frdf = frdf.sort_values(by="state")
        state_values = list(s_frdf["state"].unique())
        if 'CA' in state_values:
           default_state = state_values.index('CA')
           selected_state = st.selectbox("state", state_values, index=default_state)
        else:   
           selected_state = st.selectbox("select state", state_values)

    # with c_location:
    ffrdf = frdf[frdf['state'] == selected_state]
    s_ffrdf = ffrdf.sort_values(by='name')
    loc_values = list(s_ffrdf['name'].unique())
    if 'Madhekar Residence Home in San Diego' in loc_values:    
        default_loc = loc_values.index('Madhekar Residence Home in San Diego')
        selected_location = st.sidebar.selectbox('description', s_ffrdf['name'].unique(), index=default_loc)  
    else:
        selected_location = st.sidebar.selectbox('description', s_ffrdf['name'].unique())      
    
    # with c_selected:
    #     #st.header(f"**{selected_country} :: {selected_state} :: {selected_location}**")
    st.sidebar.subheader(f"**{selected_country} :: {selected_state} :: {selected_location}**")
    return (s_ffrdf[s_ffrdf['name'] == selected_location].iloc[0])


def save_metadata( mmp, mmf, mmef):
    st.session_state.df.to_csv(os.path.join(mmp, mmf), sep=",",index=True)

    if os.path.exists(mmp):
        st.session_state.edited_image_attributes.to_csv(os.path.join(mmp, mmef), mode='a', index=False, header=False)
    else:
        os.makedirs(mmp)
        st.session_state.edited_image_attributes.to_csv(os.path.join(mmp, mmef), index=False, header=False)   

    st.session_state.edited_image_attributes = st.session_state.edited_image_attributes.head(0)

@st.fragment
def showMap(hlat, hlon):
    with st.container(border=False):

        m = fl.Map(location=[hlat, hlon], zoom_start=4, min_zoom=3, max_zoom=10)

        fg = fl.FeatureGroup(name="zesha")

        for marker in st.session_state["markers"]:
            fg.add_child(marker)

        m.add_child(fl.LatLngPopup())
        
        map = st_folium(m, width="100%", feature_group_to_add=fg)

#@st.fragment
def editLocations(files, page, batch_size, row_size):        
    sindex = select_location_by_country_and_state(st.session_state.df_loc)  
    
    batch = files[(page - 1) * batch_size : page * batch_size]
    grid = st.columns(row_size, gap="small", vertical_alignment="top")
    col = 0
    clear_markers()
    
    for image in batch:
        with grid[col]:
            c1, c2 = st.columns([1.0, 1.0], gap="small", vertical_alignment="top")
            #print(image)
            st.session_state.df.reset_index()
            lat = st.session_state.df.at[image, "GPSLatitude"]
            lon = st.session_state.df.at[image, "GPSLongitude"]
            dt = st.session_state.df.at[image, "DateTimeOriginal"]
            label = os.path.basename(image)

            if lat != "-" and lon != '-':
                lat = round(float(st.session_state.df.at[image, "GPSLatitude"]), 6)
                lon = round(float(st.session_state.df.at[image, "GPSLongitude"]), 6)
                c2.empty()
                c2.text_input(value=lat, label=f"Lat_{image}", label_visibility="collapsed")  
                c2.empty()
                c2.text_input(value=lon, label=f"Lon_{image}", label_visibility="collapsed") 
                c2.empty()
                c2.text_input(value=dt,label=f"dt_{image}", label_visibility="collapsed", on_change=update_all_datetime_changes, key=f"dt_{image}", args=(image, 'dt'))
                add_marker(lat, lon, label, image)
            else:
                clk = c2.checkbox(label=f"location_{image}", label_visibility="collapsed")
                if clk:
                    update_latitude_longitude(image, sindex['latitude'], sindex['longitude'], sindex['name'])
                    n_row = pd.Series({"SourceFile": image, "GPSLatitude": sindex["latitude"], "GPSLongitude": sindex['longitude'], "DateTimeOriginal": dt, 'name': sindex['name']})
                    #print(n_row)
                    st.session_state.edited_image_attributes = pd.concat([st.session_state.edited_image_attributes, pd.DataFrame([n_row], columns=n_row.index)]).reset_index(drop=True)
                c2.text("")
                c2.text("")
                c2.text("")
                c2.text("")
                c2.text("")
                # r = c2.selectbox(label=f"location_{image}", label_visibility="collapsed",  options=st.session_state.df_loc.name.values, index=None, on_change=update_all_latlon())
                # if r:
                #     st.session_state["updated_location_list"].append((image, col, r))
                c2.text_input(value=dt,label=f"dt_{image}", label_visibility="collapsed", on_change=update_all_datetime_changes, key=f"dt_{image}", args=(image, 'dt')) 
                
            image = Image.open(image)  
            image.thumbnail((200,200), Image.Resampling.LANCZOS)
            c1.image(image, caption=label, output_format="JPG")
            # if lat != "-":
            #     add_marker(lat, lon, label, image)
            st.divider()    

        col = (col + 1) % row_size

"""
metadata:
  raw_data_path: /data/raw-data/
  static_metadata_path: /data/app-data/static-metadata/
  static_metadata_file: static_locations.parquet
  missing_metadata_path: /data/input-data/error/img/missing-data/
  missing_metadata_file: missing-metadata-wip.csv
  missing_metadata_filter_file: missing-metadata-filter-wip.csv
  missing_metadata_edit_file: missing-matadata-edits.csv
  home_latitude: 32.968700
  home_longitude: -117.184200
  batch_size_max: 250
  row_size_preset: 7
"""
def execute():

    (rdp, smp, smf, mmp, mmf, mmff, mmef, hlat, hlon, bsmx, rszp) = get_env()

    if 'global_user_source' not in st.session_state:
        st.session_state['global_user_source'] = ""

    st.sidebar.subheader("Storage Source", divider="gray")
    user_source_selected = st.sidebar.selectbox(
        "data source folder",
        options=ss.extract_user_raw_data_folders(rdp),
        label_visibility="collapsed",
        #on_change=remove_initialization_on_device_change()
    )
    print('---->', user_source_selected)
    if st.session_state['global_user_source'] != user_source_selected:
        print(f"--->:{st.session_state['global_user_source']} : {user_source_selected}")
        st.session_state["global_user_source"] = user_source_selected
        #print(f"--->:{user_device} : {user_source_selected}")
        remove_initialization_on_device_change()

    show_missing = st.sidebar.checkbox(label='missing metadata images')

    mmfile = mmff if show_missing else mmf

    initialize(smp, smf, mmp, mmfile, mmef, hlat, hlon, user_source_selected)

    files = pd.read_csv(os.path.join(mmp, user_source_selected, mmfile))['SourceFile']
    
    st.sidebar.subheader("Display Criteria",divider="gray")

    cb,cr,cp = st.sidebar.columns([1,1,1])
    with cb:
        batch_size = st.select_slider("Batch Size:", range(10,int(bsmx), 10))
    with cr:   
        row_size = st.select_slider("Row Size:", range(1, 10), value= int(rszp))   
        num_batches = ceil(len(files) / batch_size)
    with cp:   
        page = st.selectbox("Page Number:", range(1, num_batches + 1))

    st.sidebar.subheader('Edited Images', divider="gray")
    config = {
        'SourceFile' : st.column_config.TextColumn('image', width='small', required=True),       
        'GPSLatitude' : st.column_config.NumberColumn('latitude', min_value=-90.0, max_value=90.0, required=True),
        'GPSLongitude' : st.column_config.NumberColumn('longitude',min_value=-180.0, max_value= 180.0, required=True),
        'DateTimeOriginal' : st.column_config.TextColumn('datetime', width="small", required=False)}
    
    st.session_state["edited_image_attributes"] = st.sidebar.data_editor(st.session_state["edited_image_attributes"], column_config=config, num_rows="dynamic", use_container_width=True, height=350, hide_index=True) #

    save_btn = st.sidebar.button(label="Save: Image Metadata",  use_container_width=True) 
    if save_btn:
        save_metadata(os.path.join(mmp, user_source_selected), mmf, mmef)

    showMap(hlat=hlat, hlon=hlon)
    # with st.container(border=False):

    #     m = fl.Map(location=[hlat, hlon], zoom_start=4, min_zoom=3, max_zoom=10)

    #     fg = fl.FeatureGroup(name="zesha")

    #     for marker in st.session_state["markers"]:
    #         fg.add_child(marker)

    #     m.add_child(fl.LatLngPopup())
        
    #     map = st_folium(m, width="100%", feature_group_to_add=fg)

    # data = None
    # if map.get("last_clicked"):
    #     data = (map["last_clicked"]["latitude"], map["last_clicked"]["longitude"])

    # if data is not None:
    #     st.session_state.editor_audit_msg.append(data)
    #...

    # Location to Apply to many images
    st.subheader("Edit Location and Date", divider='gray') 

    st.sidebar.subheader('Locations', divider='gray')

    if len(files) > 0:
       editLocations(files, page, batch_size, row_size)
    else:
       st.info('No missing metadata images found!')    

    # sindex = select_location_by_country_and_state(st.session_state.df_loc)  

    # batch = files[(page - 1) * batch_size : page * batch_size]
    # grid = st.columns(row_size, gap="small", vertical_alignment="top")
    # col = 0
    # clear_markers()
    
    # for image in batch:
    #     with grid[col]:
    #         c1, c2 = st.columns([1.0, 1.0], gap="small", vertical_alignment="top")
    #         #print(image)
    #         st.session_state.df.reset_index()
    #         lat = st.session_state.df.at[image, "GPSLatitude"]
    #         lon = st.session_state.df.at[image, "GPSLongitude"]
    #         dt = st.session_state.df.at[image, "DateTimeOriginal"]
    #         label = os.path.basename(image)

    #         if lat != "-" and lon != '-':
    #             lat = round(float(st.session_state.df.at[image, "GPSLatitude"]), 6)
    #             lon = round(float(st.session_state.df.at[image, "GPSLongitude"]), 6)
    #             c2.empty()
    #             c2.text_input(value=lat, label=f"Lat_{image}", label_visibility="collapsed")  
    #             c2.empty()
    #             c2.text_input(value=lon, label=f"Lon_{image}", label_visibility="collapsed") 
    #             c2.empty()
    #             c2.text_input(value=dt,label=f"dt_{image}", label_visibility="collapsed", on_change=update_all_datetime_changes, key=f"dt_{image}", args=(image, 'dt'))
    #             add_marker(lat, lon, label, image)
    #         else:
    #             clk = c2.checkbox(label=f"location_{image}", label_visibility="collapsed")
    #             if clk:
    #                 update_latitude_longitude(image, sindex['latitude'], sindex['longitude'], sindex['name'])
    #                 n_row = pd.Series({"SourceFile": image, "GPSLatitude": sindex["latitude"], "GPSLongitude": sindex['longitude'], "DateTimeOriginal": dt, 'name': sindex['name']})
    #                 #print(n_row)
    #                 st.session_state.edited_image_attributes = pd.concat([st.session_state.edited_image_attributes, pd.DataFrame([n_row], columns=n_row.index)]).reset_index(drop=True)
    #             c2.text("")
    #             c2.text("")
    #             c2.text("")
    #             c2.text("")
    #             c2.text("")
    #             # r = c2.selectbox(label=f"location_{image}", label_visibility="collapsed",  options=st.session_state.df_loc.name.values, index=None, on_change=update_all_latlon())
    #             # if r:
    #             #     st.session_state["updated_location_list"].append((image, col, r))
    #             c2.text_input(value=dt,label=f"dt_{image}", label_visibility="collapsed", on_change=update_all_datetime_changes, key=f"dt_{image}", args=(image, 'dt')) 
                
    #         image = Image.open(image)  
    #         image.thumbnail((200,200), Image.Resampling.LANCZOS)
    #         c1.image(image, caption=label, output_format="JPG")
    #         # if lat != "-":
    #         #     add_marker(lat, lon, label, image)
    #         st.divider()    

    #     col = (col + 1) % row_size

if __name__ == "__main__":
    execute()