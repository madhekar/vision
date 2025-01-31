import streamlit as st

import os
from utils.config_util import config
import streamlit as st




def execute(source_name):
    (
        static_metadata_data_path,
        static_metadata_file,
    ) = config.dataload_config_load()

    """
      paths to import files
    """
    mdatapath = os.path.join(static_metadata_data_path, static_metadata_file)


if __name__ == "__main__":
    execute()