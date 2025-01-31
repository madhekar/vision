import os
from utils.config_util import config
from utils.util import fast_parquet_util as fpu
import streamlit as st

def transform_and_add_static_metadata(location_metadata_path, final_storage):
    pass


def execute(source_name):

    (location_metadata_path, address_metadata_path, static_metadata_path, static_metadata_file)  = config.static_metadata_config_load()

    """
      paths to import files
    """
    metadata_storage_path = os.path.join(static_metadata_path, static_metadata_file)
    transform_and_add_static_metadata(location_metadata_path=location_metadata_path, final_storage=metadata_storage_path)

if __name__ == "__main__":
    execute()