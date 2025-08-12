import site

from PyInstaller.utils.hooks import (
    collect_data_files, collect_submodules, copy_metadata
)

site_package_dir = site.getsitepackages()[0]
datas = [(f"{site_package_dir}/streamlit/runtime", "./streamlit/runtime")]
datas += copy_metadata("streamlit")
datas += collect_data_files("streamlit")
hiddenimports = collect_submodules("streamlit")