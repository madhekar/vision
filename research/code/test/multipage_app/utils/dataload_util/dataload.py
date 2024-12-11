import os
import shutil
import uuid
from utils.config_util import config
import streamlit as st
import util
from utils.util import adddata_util as adu
from streamlit_tree_select import tree_select
from utils.util import model_util as mu
from utils.util import storage_stat as ss
from utils.util import file_type_ext as fte


def path_encode(spath):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, spath))


## possible performance issue 
def copy_files_only(src_dir, dest_dir):
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir, ignore_errors=True)
        # making the destination directory
        os.makedirs(dest_dir)

    for root, dirnames, items in os.walk(src_dir):
        if not dirnames:
            if len(items) > 0:
                print(root + " - " + str(dirnames) + " - " + str(items))
                img_items = [f for f in items if os.path.splitext(f)[1].lower() in fte.image_types]
                vid_items = [f for f in items if os.path.splitext(f)[1].lower() in fte.video_types]
                txt_items = [f for f in items if os.path.splitext(f)[1].lower() in fte.document_types]
                adu_items = [f for f in items if os.path.splitext(f)[1].lower() in fte.audio_types]
                nmd_items = [f for f in items if os.path.splitext(f)[1].lower() in fte.non_media_types]
                if len(items) > 0:
                    print(root + " - " + str(dirnames) + " - " + str(items))
                    uuid_path = path_encode(root)
                    f_dest = os.path.join(dest_dir, uuid_path)
                    print(src_dir + " - " + f_dest + " - " + str(len(items)))
                    os.makedirs(f_dest)
                    for item in items:
                        item_path = os.path.join(root, item)
                        print(item_path + " -> " + dest_dir)
                        if os.path.isfile(item_path):
                            print("**" + item_path + " - " + dest_dir)
                            try:
                                shutil.copy(item_path, f_dest)
                            except FileNotFoundError:
                                print("Source file not found.")
                            except PermissionError:
                                print("Permission denied.")
                            except FileExistsError:
                                print("Destination file already exists.")
                            except Exception as e:
                                print(f"An error occurred: {e}")

def execute():
    (
        raw_data_path,
        input_image_path,
        input_video_path,
        input_txt_path,
    ) = config.dataload_config_load()

    """
    select data source to trim data
    """
    # source_list = []
    # # source_list = get_external_devices(get_user())
    # source_list = mu.extract_user_raw_data_folders(raw_data_path)
    # if len(source_list) > 0:
    #     ext = st.sidebar.selectbox(label="Select Source", options=source_list)

    # st.sidebar.caption("CHECK FOLDERS TO TRIM", unsafe_allow_html=True)
    # display_folder_tree(get_path_as_dict(os.path.join(raw_data_path, ext)))
    # st.sidebar.button(label="TRIM CHECKED FOLDERS", use_container_width=True)
    # # c1.text_area(label="External Source Structure", value= display_tree(os.path.join('/media/madhekar/' , ext)))


if __name__ == "__main__":
    execute()