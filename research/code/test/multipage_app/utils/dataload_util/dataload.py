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

def clean_media_folders(folder):
    # create clean destination folders
    if os.path.exists(folder):
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder)

def handle_copy_media_files(root, fdest_media, uuid_path, media_items):

    #print(root + " - " + str(dirnames) + " - " + str(media_items))
    f_dest = os.path.join(fdest_media, uuid_path)
    #print(src_dir + " - " + f_dest + " - " + str(len(media_items)))
    os.makedirs(f_dest)
    for item in media_items:
        item_path = os.path.join(root, item)
        print(item_path + " -> " + f_dest)
        if os.path.isfile(item_path):
            print("**" + item_path + " - " + fdest_media)
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

## possible performance issue 
def copy_files_only(src_dir, fdest_image, fdest_txt, fdest_video, fdest_audio ):

    img_items, txt_items, vid_items, adu_items = ([] for i in range(4))

    #create clean destination folders
    clean_media_folders(fdest_image)
    clean_media_folders(fdest_txt)
    clean_media_folders(fdest_video)
    clean_media_folders(fdest_audio)

    for root, dirnames, items in os.walk(src_dir):
        if not dirnames:
            if len(items) > 0:
                print(root + " - " + str(dirnames) + " - " + str(items))
                img_items = [f for f in items if os.path.splitext(f)[1].lower() in fte.image_types]
                vid_items = [f for f in items if os.path.splitext(f)[1].lower() in fte.video_types]
                txt_items = [f for f in items if os.path.splitext(f)[1].lower() in fte.document_types]
                adu_items = [f for f in items if os.path.splitext(f)[1].lower() in fte.audio_types]

                #generate uuid path
                uuid_path = path_encode(root)
                
                # handle image items
                if len(img_items) > 0:
                    handle_copy_media_files(root, fdest_image, uuid_path, img_items)

                if len(vid_items) > 0:
                    handle_copy_media_files(root, fdest_video, uuid_path, vid_items)    

                if len(img_items) > 0:
                    handle_copy_media_files(root, fdest_txt, uuid_path, txt_items)

                if len(adu_items) > 0:
                    handle_copy_media_files(root, fdest_audio, uuid_path, adu_items)     
                    
                    # print(root + " - " + str(dirnames) + " - " + str(img_items))
                    # f_dest = os.path.join(fdest_image, uuid_path)
                    # print(src_dir + " - " + f_dest + " - " + str(len(img_items)))
                    # os.makedirs(f_dest)
                    # for item in img_items:
                    #     item_path = os.path.join(root, item)
                    #     print(item_path + " -> " + f_dest)
                    #     if os.path.isfile(item_path):
                    #         print("**" + item_path + " - " + fdest_image)
                    #         try:
                    #             shutil.copy(item_path, f_dest)
                    #         except FileNotFoundError:
                    #             print("Source file not found.")
                    #         except PermissionError:
                    #             print("Permission denied.")
                    #         except FileExistsError:
                    #             print("Destination file already exists.")
                    #         except Exception as e:
                    #             print(f"An error occurred: {e}")

def execute(source_name):
    (
        raw_data_path,
        input_image_path,
        input_txt_path,
        input_video_path,
        input_audio_path
    ) = config.dataload_config_load()

    """
      paths to import files
    """
    ipath = os.path.join(input_image_path, source_name)
    tpath = os.path.join(input_txt_path, source_name)
    vpath = os.path.join(input_video_path, source_name)
    apath = os.path.join(input_audio_path, source_name)

    copy_files_only(raw_data_path, ipath, tpath, vpath, apath)
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