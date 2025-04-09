import os
import pandas as pd
from utils.config_util import config
from utils.util import fast_parquet_util as fpu
from utils.util import storage_stat as ss
import user_static_loc as usl
import streamlit as st

colors = ["#ae5a41", "#1b85b8"]
# create user specific static image metadata "locations" not found in default static metadata
def generate_user_specific_static_metadata(missing_path, missing_file, location_path, user_draft_location_metadata_path, user_draft_location_metadata_file):

    # dataframe for all default static locations
    default_df = fpu.combine_all_default_locations(location_path)

    # additional locations not included in default static locations
    df_unique =  usl.get_unique_locations(pd.read_csv(os.path.join(missing_path, missing_file)),default_df)

    #create draft static unique location file
    df_unique.to_csv(os.path.join(user_draft_location_metadata_path, user_draft_location_metadata_file), index=False, encoding="utf-8")

def transform_and_add_static_metadata(location_metadata_path, user_location_metadata,  final_parquet_storage):
    fpu.add_all_locations(location_metadata_path, user_location_metadata, final_parquet_storage)

"""
datapaths:
  raw_data_path: /home/madhekar/work/home-media-app/data/raw-data/
static-locations:
  location_metadata_path: /home/madhekar/work/home-media-app/data/static-data/static-locations/default
  user_location_metadata_path: /home/madhekar/work/home-media-app/data/static-data/static-locations/user-specific
  user_location_metadata_file: user-specific.csv
  user_draft_location_metadata_path_ext: draft
  user_draft_locations_metadata_file: user-specific-draft.csv

  missing_metadata_path: /home/madhekar/work/home-media-app/data/input-data-1/error/img/missing-data
  missing_metadata_file: missing-metadata-wip.csv

  static_metadata_path: /home/madhekar/work/home-media-app/data/app-data/static-metadata
  static_metadata_file: static_locations.parquet

"""
def execute():
    (
        raw_data_path,
        location_metadata_path,

        user_location_metadata_path,
        user_location_metadata_file,

        user_draft_location_metadata_path_ext,
        user_draft_location_metadata_file,
        
        missing_metadata_path,
        missing_metadata_file,
        
        static_metadata_path,
        static_metadata_file,
    ) = config.static_metadata_config_load()

    st.sidebar.subheader("Storage Source", divider="gray")
    user_source_selected = st.sidebar.selectbox(
        "data source folder",
        options=ss.extract_user_raw_data_folders(raw_data_path),
        label_visibility="collapsed",
    )
    user_location_metadata_path =  os.path.join(user_location_metadata_path, user_source_selected)
    user_draft_location_metadata_path = os.path.join(user_location_metadata_path, user_source_selected, user_draft_location_metadata_path_ext)
    
    location_metadata_path = os.path.join(location_metadata_path, user_source_selected)

    missing_metadata_path = os.path.join(missing_metadata_path, user_source_selected)

    # paths to import static location files
    metadata_storage_path = os.path.join(static_metadata_path, user_source_selected, static_metadata_file)

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
    
    c1, c1 = st.columns([1,1], gap="small")
    with c1:
        c1_create = st.button("user specific static metadata")
        if c1_create:
            generate_user_specific_static_metadata(missing_metadata_path, missing_metadata_file, location_metadata_path, user_draft_location_metadata_path, user_draft_location_metadata_file)
    with c2:
        c2_create = st.button("final static metadata")
        if c2_create:
            # clean previous parquet
            try:
                if os.path.exists(metadata_storage_path):
                    st.warning(
                        f"cleaning previous static metadata storage: {metadata_storage_path}"
                    )
                    os.remove(metadata_storage_path)
            except Exception as e:
                st.error(f"Exception encountered wile removing metadata file: {e}")

            st.info(f"creating new static metadata storage: {metadata_storage_path}")
            transform_and_add_static_metadata(
                location_metadata_path,
                user_location_metadata_path,
                metadata_storage_path,
            )

if __name__ == "__main__":
    execute()