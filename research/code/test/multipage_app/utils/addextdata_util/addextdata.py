
import os
import getpass
import shutil
import streamlit as st
from utils.util import adddata_util as adu
from utils.util import model_util as mu
from streamlit_tree_select import tree_select
from utils.config_util import config

media_extensions = [
    ".mp3",
    ".mp4",
    ".avi",
    ".mov",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".heif",
    ".aac",
    ".wav",
    ".amv",
    ".mpeg",
    ".flv",
    ".avi",
    ".webp",
    ".tif",
    ".bmp",
    ".eps",
    '.sbd', 
    '.ra', 
    '.au', 
    '.wma', 
    '.wmv', 
    '.3gp'
]  # Add more as needed
document_extensions = [
    '.txt', 
    '.pdf', 
    '.doc', 
    '.docx', 
    '.xls', 
    '.xlsx', 
    '.ppt', 
    '.pptx', 
    '.log', 
    '.java', 
    '.c', 
    '.py', 
    '.js', 
    '.html', 
    '.asp', 
    '.css', 
    '.xps', 
    '.rtf', 
    '.csv', 
    '.wps', 
    '.msg' , 
    '.dta', 
    '.swift', 
    '.pl', 
    '.sh', 
    '.bat', 
    '.ts', 
    '.cpp']  # Add more as needed

# nodes=[]
# nodes.append(adu.path_dict("/home/madhekar/work/home-media-app/data/raw-data"))

# con = st.sidebar.container(height=500)
# with con:
#     return_select = tree_select(nodes, no_cascade=True)

# con1 = st.sidebar.expander(label="checked folders")  # container(height=500)
# with con1:
#     selected = []
#     for e in return_select["checked"]:
#         e0 = e.split("@@")[0]
#         selected.append(e0)
#         st.write(e0)

# st.sidebar.button(label="Trim(Checked-Folders)")

def get_user():
    return getpass.getuser()

def get_external_devices(user):
    return os.listdir(f"/media/{user}")

def get_existing_data_sources(dpath):
     pass
     
def create_external_data_path(user, selected_device):
    return f"/media/{user}/{selected_device}"   
  
# def overrite(data_source):
#     st.write(f'Are Sure, Overrite Existing {data_source} Data Source')
#     if st.button('Submit'):
#         st.session_state.overrite = 

def deep_copy_external_drive_to_raw(src, dest):
    if os.path.exists(dest):
        shutil.rmtree(dest, ignore_errors=True)
        # making the destination directory
    #     os.makedirs(dest)
    # else:
    #     os.makedire(dest)
    try:
        shutil.copytree(src, dest)
    except FileNotFoundError as e:
        st.error(f'error: folder {src} not found occured {e}')    
    except Exception as e:
        st.error(f'An error occured: {e}')    

def copy_folders_with_files(src_dir, dest_dir):
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir, ignore_errors=True)
        # making the destination directory
        os.makedirs(dest_dir)

    for root, dirnames, items in os.walk(src_dir):
        if not dirnames:
            if len(items) > 0:
                print(root + " - " + str(dirnames) + " - " + str(items))
                items = [
                    f
                    for f in items
                    if os.path.splitext(f)[1].lower() in media_extensions
                    or os.path.splitext(f)[1].lower() in document_extensions
                ]
                if len(items) > 0:
                    print(root + " - " + str(dirnames) + " - " + str(items))
                    uuid_path = mu.create_uuid_from_string(root)
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

"""
raw_data_path, input_image_path, input_txt_path, input_video_path, input_audio_path
"""

def execute():
    (
        raw_data_path,
        input_image_path,
        input_txt_path,
        input_video_path,
        input_audio_path
        
    ) = config.dataload_config_load()

    # select load data from external data source such as USB device
    user = get_user()

    devices_list = get_external_devices(user=user)

    #show external devices in dropdown box
    ext_source = st.sidebar.selectbox(label="EXTERNAL DATA SOURCES", options=devices_list)
    
    #show data already imported data sources
    source_list = mu.extract_user_raw_data_folders(raw_data_path)
    s=""
    for str in source_list:
        s += str + '\n'

    if len(source_list) > 0:
       #st.sidebar.selectbox(label="Existing Imported Data Sources", options=source_list)
       st.sidebar.text_area(label="EXISTING DATA SOURCES", value=s, height=100)

    # create new path string by appending raw-data path with source-name from external device
    if ext_source in source_list:
        ans = st.sidebar.toggle(f'DO YOU REALLY LIKE TO OVERWRITE ON [**{ext_source}**] DATA SOURCE?')
        if ans:   
            st.sidebar.button(label="IMPORT & OVERWRIDE DATA", use_container_width=True)
            # remove folders and files
            # import folders and files - generate file paths based on uuid.uuid5
            # keep base same
            #deep_copy_external_drive_to_raw(create_external_data_path(get_user(), ext_source),os.path.join(raw_data_path, ext_source))
            copy_folders_with_files(create_external_data_path(get_user(), ext_source),os.path.join(raw_data_path, ext_source))
    else:
        st.sidebar.button(label="IMPORT DATA", use_container_width=True)
        # import folders and files - generate file paths based on uuid.uuid5
        # keep base same
        copy_folders_with_files(create_external_data_path(get_user(), ext_source),os.path.join(raw_data_path, ext_source))
        #deep_copy_external_drive_to_raw(create_external_data_path(get_user(), ext_source),os.path.join(raw_data_path, ext_source))

if __name__ == "__main__":
    execute()