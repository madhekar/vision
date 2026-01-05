import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from utils.config_util import config
from utils.util import fast_parquet_util as fpu
from utils.util import storage_stat as ss
from utils.static_metadata_load_util import user_static_loc as usl
from utils.preprocess_util import preprocess as pp
from utils.face_util import base_face_train as bft_train
from utils.face_detection_util import face_trainer as ft
from utils.filter_util import filter as fu
from utils.filter_util import torch_filter as tfu
from utils.face_util import base_face_predict as bft_predict
import streamlit as st
import altair as alt
from utils.util import folder_chart as fc
import plotly.express as px

colors = ["#ae5a41", "#1b85b8"]
# create user specific static image metadata "locations" not found in default static metadata
def generate_user_specific_static_metadata(missing_path, missing_file, location_path, user_location_metadata_path, user_location_metadata_file):

    # dataframe for all default static locations
    default_df = fpu.combine_all_default_locations(location_path)

    # additional locations not included in default static locations
    df_unique =  usl.get_unique_locations(pd.read_csv(os.path.join(missing_path, missing_file)),default_df)

    #create draft static unique location file
    df_unique.to_csv(os.path.join(user_location_metadata_path, user_location_metadata_file), index=False, encoding="utf-8")
    #print(f'---->{df_unique} : {len(df_unique)}')

def transform_and_add_static_metadata(location_metadata_path, user_location_metadata, user_location_metadata_file, final_parquet_storage):
    fpu.add_all_locations(location_metadata_path, user_location_metadata, user_location_metadata_file, final_parquet_storage)

"""
datapaths:
  raw_data_path: /home/madhekar/work/home-media-app/data/raw-data/
static-metadata:
  static_metadata_path: /home/madhekar/work/home-media-app/data/app-data/static-metadata  
static-faces: 
  faces_metadata_path: /home/madhekar/work/home-media-app/data/app-data/static-metadata/faces
static-locations:
  default_location_metadata_path: /home/madhekar/work/home-media-app/data/app-data/static-metadata/locations/default
  user_location_metadata_path: /home/madhekar/work/home-media-app/data/app-data/static-metadata/locations/user-specific
  user_location_metadata_file: user-specific.csv
  final_user_location_metadata_file: static_locations.parquet
missing-metadata:  
  missing_metadata_path: /home/madhekar/work/home-media-app/data/input-data/error/img/missing-data
  missing_metadata_file: missing-metadata-wip.csv
  missing_metadata_filter_file: missing-metadata-filter-wip.csv
 
            (
            raw_data_path, 

            static_metadata_path,

            faces_metadata_path, 

            default_location_metadata_path, 
            user_location_metadata_path, 
            user_location_metadata_file, 
            final_user_location_metadata_file, 

            missing_metadata_path,
            missing_metadata_file,
            missing_metadata_filter_file
            )
            
"""
def execute():
    (
        raw_data_path,
        static_metadata_path,
        faces_metadata_path,
        filter_metadata_path,
        default_location_metadata_path,
        user_location_metadata_path,
        user_location_metadata_file,
        final_user_location_metadata_file,
        missing_metadata_path,
        missing_metadata_file,
        missing_metadata_filter_file,
    ) = config.static_metadata_config_load()

    st.sidebar.subheader("Storage Source", divider="gray")
    user_source_selected = st.sidebar.selectbox(
        "data source folder",
        options=ss.extract_user_raw_data_folders(raw_data_path),
        label_visibility="collapsed",
    )
    
    user_location_metadata_path =  os.path.join(user_location_metadata_path, user_source_selected)

    if not os.path.exists(user_location_metadata_path):
        os.makedirs(user_location_metadata_path, exist_ok=True)

    missing_metadata_path = os.path.join(missing_metadata_path, user_source_selected)

    # paths to import static location files
    final_user_metadata_storage_path = os.path.join(user_location_metadata_path, final_user_location_metadata_file)

    c1, c2, c3, c4 = st.columns([.3, .3, .3, .3], gap="medium")
 
    with c1:
        st.subheader("Static Metadata", divider='gray')

        dfs = ss.extract_all_file_stats_in_folder(static_metadata_path)
        dfs = dfs.reset_index(names="file_type")
        dfs['size'] = dfs['size'].apply(lambda x: x /(pow(1024, 2))).astype(float)
        dfs['connt'] = dfs['count'].astype(int)

        base = alt.Chart(dfs).encode(
        y=alt.Y('file_type:N', sort='-x', title='File Type'), # Sort descending by x-value
        color=alt.Color("file_type:N", scale=alt.Scale( scheme='dark2')),
        tooltip=['file_type', 'count', 'size']
        ).properties(
           title='File count and Size by Type',
        )

        # Bar chart for Size (MB)
        size_chart = base.mark_bar(color='skyblue', opacity=0.7).encode(
            x=alt.X('size:Q', title='Total Size GB'),
        )

        # Text labels for count on the bars
        text_count = size_chart.mark_text(
            align='left',
            baseline='middle',
            dx=3 # Nudges text to the right of the bar
        ).encode(
            x='size:Q',
            text='count:Q',
            color=alt.value('black')
        )

        # Combine the bar chart and text labels
        chart = size_chart + text_count
        st.altair_chart(chart, use_container_width=True)
   
    with c2:
        dfl = ss.extract_all_file_stats_in_folder(default_location_metadata_path)
        dfa = ss.extract_all_file_stats_in_folder(user_location_metadata_path) 
        print('--->', dfa.head())
        count = len(dfa) if len(dfa) > 0 else 0    
        size = round(sum(dfa["size"])/(pow(1024,2)),2) if len(dfa) >0 else 0  
        print(dfl['count'], dfa['count'])
        c2a, c2b = st.columns([1,1], gap="small")
        with c2a:
            st.subheader("Locations", divider='gray')
            st.metric("Number of location files", sum(dfl['count']))
            st.metric("Total size of location files (MB)", round(dfl["size"]/(pow(1024,2)), 2),delta=.23)
        with c2b:
            st.subheader("User Locations", divider='gray')
            st.metric("Number of user location files", int(count))
            st.metric("Total size of user locations files (MB)",  int(size))

    with c3:
        st.subheader('Number of Images / Person', divider='gray') 
        df = fc.sub_file_count( faces_metadata_path) #"/home/madhekar/work/home-media-app/data/app-data/static-metadata/faces")
        print(f'++++{df}')
        st.bar_chart(df, x="person", y="number of images", color=["#1b85b8"], horizontal=True)

    with c4:
        st.subheader('Image Classifier Filter', divider='gray')    
        df = fc.sub_file_count(filter_metadata_path)
        st.bar_chart(df, x="person", y="number of images", color=["#1b85b8"], horizontal=True)

    st.divider()
 
    ca, cb, cc, cd = st.columns([0.3, 0.3, 0.3, 0.3], gap="small", vertical_alignment="top")
   
    with ca:            
        ca_create = st.button("**user specific locations**", use_container_width=True, type="primary")
        ca_status = st.status('create user specific locations', state="running", expanded=True)
        with ca_status:
            if ca_create:
                ca.info(f'starting to create user specific static location data for: {user_source_selected}')
                print(user_location_metadata_file)
                if not os.path.exists(os.path.join(user_location_metadata_path, user_location_metadata_file)):
                   generate_user_specific_static_metadata(missing_metadata_path, missing_metadata_file, default_location_metadata_path, user_location_metadata_path, user_location_metadata_file) 
                ca_status.update(label='user specific locations complete!', state='complete', expanded=False)
    with cb:
        cb_metadata = st.button("**aggregate all locations**", use_container_width=True, type="primary")
  
        cb_status = st.status('create static location store', state='running', expanded=True)
        with cb_status:
            if cb_metadata:
                ca.info("starting to create total static location data. ")
                # clean previous parquet
                try:
                    if os.path.exists(final_user_metadata_storage_path):
                        st.warning(f"cleaning previous static metadata storage: {final_user_metadata_storage_path}")
                        os.remove(final_user_metadata_storage_path)
                except Exception as e:
                    st.error(f"Exception encountered wile removing metadata file: {e}")

                st.info(f"creating new static metadata storage: {final_user_metadata_storage_path}")
                transform_and_add_static_metadata( default_location_metadata_path, user_location_metadata_path, user_location_metadata_file, final_user_metadata_storage_path)
                cb_status.update(label="metadata creation complete!", state="complete", expanded=False)  
    with cc:
        cc_metadata = st.button("**Refresh people detection model**", use_container_width=True, type="primary")
        cc_status = st.status('create people names ', state='running', expanded=True)  
        with cc_status:
            if cc_metadata:
                    cc_status.info("starting to create face model.")
                    st.info('step: - 1: train know faces for search...')
                    ft.execute()
                    cc_status.update(label="face detection model complete!", state="complete", expanded=False) 

    with cd:
        cd_filter = st.button("**Refresh image filter model**", use_container_width=True, type="primary")
        cd_status = st.status('filter image model ', state='running', expanded=True)  
        with cd_status:
            if cd_filter:
                    cd_status.info("starting to image filter model.")
                    st.info('step: - 1: train image filter for search...')
                    ys, yt = tfu.execute()
                    print(classification_report(ys, yt))
                    cd_status.info(classification_report(ys,yt))
                    cd_status.update(label="Image filter model complete!", state="complete", expanded=False)             

if __name__ == "__main__":
    execute()