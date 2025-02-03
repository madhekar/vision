import os
from utils.config_util import config
from utils.util import fast_parquet_util as fpu
from utils.util import storage_stat as ss
import streamlit as st

colors = ["#ae5a41", "#1b85b8"]
def transform_and_add_static_metadata(location_metadata_path, final_parquet_storage):
    fpu.add_all_locations(location_metadata_path, final_parquet_storage)


def execute():
    (location_metadata_path, address_metadata_path, static_metadata_path, static_metadata_file)  = config.static_metadata_config_load()
    # paths to import static location files
    metadata_storage_path = os.path.join(static_metadata_path, static_metadata_file)


    c1, c2 = st.columns([0.5, 1], gap="medium")
    with c1:
        st.subheader("Static Metadata")
        dfs = ss.extract_all_file_stats_in_folder(static_metadata_path)
        st.bar_chart(
            dfs,
            horizontal=False,
            stack=True,
            y_label='total size(MB) & count of files',
            use_container_width=True,
            color=colors
        )
    with c2:
        st.subheader("Raw Static Metadata")

        dfl = ss.extract_all_file_stats_in_folder(location_metadata_path)
        dfa = ss.extract_all_file_stats_in_folder(address_metadata_path)        
        print(dfl, dfa)
        c2a, c2b = st.columns([1,1], gap="small")
        with c2a:
            st.subheader('locations')
            st.bar_chart(
                dfl,
                horizontal=False,
                stack=True,
                y_label="total size(MB) & count of files",
                use_container_width=True,
                color=colors,
            )
        with c2b:
            st.subheader('addresses')
            st.bar_chart(
                dfa,
                horizontal=False,
                stack=True,
                y_label="total size(MB) & count of files",
                use_container_width=True,
                color=colors,
            )  
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
        transform_and_add_static_metadata(location_metadata_path, metadata_storage_path)

if __name__ == "__main__":
    execute()