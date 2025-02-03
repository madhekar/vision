import os
from utils.config_util import config
from utils.util import fast_parquet_util as fpu
import streamlit as st

def transform_and_add_static_metadata(location_metadata_path, final_parquet_storage):
    fpu.add_all_locations(location_metadata_path, final_parquet_storage)


def execute():
    (location_metadata_path, address_metadata_path, static_metadata_path, static_metadata_file)  = config.static_metadata_config_load()

    # paths to import static location files
    metadata_storage_path = os.path.join(static_metadata_path, static_metadata_file)
    transform_and_add_static_metadata(location_metadata_path, metadata_storage_path)

if __name__ == "__main__":
    execute()