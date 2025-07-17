import os
import numpy as np
import pandas as pd
from utils.config_util import config
from utils.util import fast_parquet_util as fpu
from utils.util import storage_stat as ss
from utils.static_metadata_load_util import user_static_loc as usl
from utils.preprocess_util import preprocess as pp
from utils.face_util import base_face_train as bft_train
from utils.face_util import base_face_predict as bft_predict
import streamlit as st
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
  missing_metadata_path: /home/madhekar/work/home-media-app/data/input-data-1/error/img/missing-data
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

    c1, c2, c3 = st.columns([.5, .3, .5], gap="medium")
    with c1:
        st.subheader("Static Metadata")
        c11,c12 = c1.columns([1,1])
        dfs = ss.extract_all_file_stats_in_folder(static_metadata_path)
        dfs['size'] = dfs['size'].apply(lambda x: x /(pow(1024, 2)))
        #dfs['count'] = dfs['size'].apply(lambda x: x /10)
        with c11:
           st.bar_chart(dfs, y="count", color=["#1b85b8"], horizontal=True, x_label= "Count")
        with c12:   
           st.bar_chart(dfs, y="size", color=["#ae5a41"], horizontal=True, x_label= "Size MB")
   
    with c2:
        dfl = ss.extract_all_file_stats_in_folder(default_location_metadata_path)
        dfa = ss.extract_all_file_stats_in_folder(user_location_metadata_path) 
        print('--->', dfa.head())
        count = len(dfa) if len(dfa) > 0 else 0    
        size = round(sum(dfa["size"])/(pow(1024,2)),2) if len(dfa) >0 else 0  
        print(dfl['count'], dfa['count'])
        c2a, c2b = st.columns([1,1], gap="small")
        with c2a:
            st.subheader("Locations")
            st.metric("Number of location files", sum(dfl['count']))
            st.metric("Total size of location files (MB)", round(dfl["size"]/(pow(1024,2)), 2),delta=.23)
        with c2b:
            st.subheader("User Locations")
            st.metric("Number of user location files", int(count))
            st.metric("Total size of user locations files (MB)",  int(size))

    with c3:
        st.subheader('Number of Images / Person') 
        df = fc.sub_file_count( faces_metadata_path) #"/home/madhekar/work/home-media-app/data/app-data/static-metadata/faces")
        st.bar_chart(df, x="person", y="number of images", color=["#c3cb71"], horizontal=True)
    st.divider()
 
    ca, cb, cc = st.columns([0.4, 0.4, 0.5], gap="small", vertical_alignment="top")
   
    with ca:            
        ca_create = st.button("**user specific locations**", use_container_width=True)
        ca_status = st.status('create user specific locations', state="running", expanded=True)
        with ca_status:
            if ca_create:
                ca.info('starting to create user specific static location data.')
                print(user_location_metadata_file)
                if not os.path.exists(os.path.join(user_location_metadata_path, user_location_metadata_file)):
                   generate_user_specific_static_metadata(missing_metadata_path, missing_metadata_file, default_location_metadata_path, user_location_metadata_path, user_location_metadata_file) 
                ca_status.update(label='user specific locations complete!', state='complete', expanded=False)
    with cb:
        cb_metadata = st.button("**aggregate all locations**", use_container_width=True)
  
        cb_status = st.status('create static location store', state='running', expanded=True)
        with cb_status:
            if cb_metadata:
                ca.info("starting to create total static location data.")
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
        cc_metadata = st.button("**Refresh people detection model**", use_container_width=True)
        cc_status = st.status('create people names ', state='running', expanded=True)  
        with cc_status:
            if cc_metadata:
                    cc_status.info("starting to create face model.")
                    st.info('step: - 1: train know faces for search...')
                    bft_train.exec()
                    cc_status.update(label="face detection model complete!", state="complete", expanded=False) 
if __name__ == "__main__":
    execute()