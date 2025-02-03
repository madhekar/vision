import os
from utils.config_util import config
from utils.util import fast_parquet_util as fpu
from utils.util import storage_stat as ss
import streamlit as st

def transform_and_add_static_metadata(location_metadata_path, final_parquet_storage):
    fpu.add_all_locations(location_metadata_path, final_parquet_storage)


def execute():
    (location_metadata_path, address_metadata_path, static_metadata_path, static_metadata_file)  = config.static_metadata_config_load()
    # paths to import static location files
    metadata_storage_path = os.path.join(static_metadata_path, static_metadata_file)


    c1, c2 = st.columns([0.5, 1], gap="medium")
    with c1:
        st.caption("Static Metadata")
        dfs = ss.extract_all_file_stats_in_folder(static_metadata_path)
        st.write(dfs)
    with c2:
        st.caption("Raw Static Metadata")
        dfl = ss.extract_all_file_stats_in_folder(location_metadata_path)
        dfa = ss.extract_all_file_stats_in_folder(address_metadata_path)
        st.write(dfl)
        st.write(dfa)

    ld = st.button("clean & load static metadata")
    if ld:
        transform_and_add_static_metadata(location_metadata_path, metadata_storage_path)

if __name__ == "__main__":
    execute()