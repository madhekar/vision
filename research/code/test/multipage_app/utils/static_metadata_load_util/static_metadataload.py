import os
from utils.config_util import config
from utils.util import fast_parquet_util as fpu
from utils.util import storage_stat as ss
import streamlit as st

colors = ["#ae5a41", "#1b85b8"]
def transform_and_add_static_metadata(location_metadata_path, user_loation_metadata,  final_parquet_storage):
    fpu.add_all_locations(location_metadata_path, user_loation_metadata,final_parquet_storage)


def execute():
    (raw_data_path, location_metadata_path, user_location_metadata_path, static_metadata_path, static_metadata_file)  = config.static_metadata_config_load()
    # paths to import static location files
    metadata_storage_path = os.path.join(static_metadata_path, static_metadata_file)
   

    st.sidebar.subheader("Storage Source", divider="gray")
    user_source_selected = st.sidebar.selectbox(
        "data source folder",
        options=ss.extract_user_raw_data_folders(raw_data_path),
        label_visibility="collapsed",
    )
    user_location_metadata_path =  os.path.join(user_location_metadata_path, user_source_selected)
    
    c1, c2 = st.columns([0.5, 1], gap="medium")
    with c1:
        st.subheader("Static Metadata")
        dfs = ss.extract_all_file_stats_in_folder(static_metadata_path)
        st.metric("Number of location files", dfs['count'])
        st.metric("Total size of location files (MB)", round(dfs["size"]/(pow(1024,2)), 2), delta=.7)
    with c2:
        dfl = ss.extract_all_file_stats_in_folder(location_metadata_path)
        dfa = ss.extract_all_file_stats_in_folder(user_location_metadata_path)        
        print(dfl['count'], dfa)
        c2a, c2b = st.columns([1,1], gap="small")
        with c2a:
            st.subheader("Locations")
            st.metric("Number of location files", dfl['count'])
            st.metric("Total size of location files (MB)", round(dfl["size"]/(pow(1024,2)), 2),delta=.23)
        with c2b:
            st.subheader("User Locations")
            st.metric("Number of user location files", dfa['count'])
            st.metric("Total size of user locations files (MB)",  round(dfa["size"]/(pow(1024,2)), 2), delta=-.1) 
    st.divider()
    ld = st.button("clean & load static metadata")
    if ld:
        #clean previous parquet
        try:
          if os.path.exists(metadata_storage_path):
             st.warning(f"cleaning previous static metadata storage: {metadata_storage_path}")
             os.remove(metadata_storage_path)
        except Exception as e:
            st.error(f"Exception encountered wile removing metadata file: {e}")
        st.info(f"creating new static metadata storage: {metadata_storage_path}")
        transform_and_add_static_metadata(location_metadata_path, user_location_metadata_path, metadata_storage_path)

if __name__ == "__main__":
    execute()