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
from utils.util import statusmsg_util as sm

def path_encode(spath):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, spath))

def clean_media_folders(folder):
    # create clean destination folders
    sm.add_messages("load", f"s|starting to clean {folder}")
    if os.path.exists(folder):
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder)
    sm.add_messages("load", f"s|done to cleaning {folder}")    

def handle_copy_media_files(root, fdest_media, uuid_path, media_items):

    #print(root + " - " + str(dirnames) + " - " + str(media_items))
    f_dest = os.path.join(fdest_media, uuid_path)
    #print(src_dir + " - " + f_dest + " - " + str(len(media_items)))
    os.makedirs(f_dest)
    sm.add_messages('load', f's|source: {root} destination: {fdest_media} coping {len(media_items)} number of files.')
    for item in media_items:
        item_path = os.path.join(root, item)
        #print(item_path + " -> " + f_dest)
        if os.path.isfile(item_path):
            #print("**" + item_path + " - " + fdest_media)
            try:
                shutil.copy(item_path, f_dest)
            except FileNotFoundError:
                e1 = ReferenceError("Source file not found.")
                sm.add_messages("load",f"e|exception: {e1} Source file not found {item_path}")
                continue
            except PermissionError:
                e2 = RuntimeError("Permission denied.")
                sm.add_messages("load",f"e|exception: {e2} file permissing denied {item_path}")
                continue
            except FileExistsError:
                e3 = RuntimeError("Destination file already exists.")
                sm.add_messages("load", f"e|exception: {e3} destination file: {f_dest} already exists.")
                continue
            except Exception as e:
                e4 = RuntimeError(f"An Unknown error occurred in dataload: {e}")
                sm.add_messages("load", f"e|exception: {e4} unknown file exception: {f_dest}.")
                continue

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

                if len(txt_items) > 0:
                    handle_copy_media_files(root, fdest_txt, uuid_path, txt_items)

                if len(adu_items) > 0:
                    handle_copy_media_files(root, fdest_audio, uuid_path, adu_items)     


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