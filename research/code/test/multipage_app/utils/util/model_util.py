
import os
import base64
import hashlib
import uuid
import glob
import datetime
import shutil
from pathlib import Path
import streamlit as st
import pandas as pd



default_home_loc = (32.968699774829794, -117.18420145463236)
default_date_time = ["2000","01","01","2000:01:01 01:01:01"] 
# def_date_time = "2000:01:01 01:01:01"

# recursive call to get all image filenames
def getRecursive(rootDir, chunk_size=10):
    f_list=[]
    
    for fn in glob.iglob(rootDir + "/**/*", recursive=True):
        if not os.path.isdir(os.path.abspath(fn)):
            f_list.append(os.path.abspath(fn))    
    for i in range(0, len(f_list), chunk_size):
        yield f_list[i:i+chunk_size]        
      
def drop_except(df, columns_to_keep):
    columns_to_drop = [col for col in df.columns if col not in columns_to_keep] 
    return df.drop(columns=columns_to_drop)    

def createProgressBar(pg_caption, pg_int_percentage, pg_colour, pg_bgcolour):
    pg_int_percentage = str(pg_int_percentage).zfill(2)
    pg_html = f"""<table style="width:50%; border-style: none;">
                        <tr style='font-weight:bold;'>
                            <td style='background-color:{pg_bgcolour};'>{pg_caption}: <span style='accent-color: {pg_colour}; bgcolor: transparent;'>
                                <progress value='{pg_int_percentage}' max='100'>{pg_int_percentage}%</progress> </span>{pg_int_percentage}% 
                            </td>
                        </tr>
                    </table><br>"""
    return pg_html

# st.markdown(createProgressBar("Positive", 62, "#A5D6A7", "#B2EBF2"), True)
# st.markdown(createProgressBar("Neutral", 40, "#FFD54F", "#B2EBF2"), True)
# st.markdown(createProgressBar("Negative", 65, "red", "#B2EBF2"), True)



# keep track of processed files in loop
def is_processed_batch(ilist, processed_df):
  rlist = []
  if processed_df is not None and not processed_df.empty:
    for file_name in ilist: 
        if file_name not in processed_df['uri'].values:
            rlist.append(file_name)   
  else:
      return ilist          
  return rlist


def img_to_base64bytes(img_path):
    with open(img_path, "rb") as f:
        data = f.read()
        return base64.b64encode(data).decode()


def generate_sha256_hash(txt):
    sha256_hash = hashlib.sha256()
    sha256_hash.update(txt.encode("utf-8"))
    return sha256_hash.hexdigest()


@st.dialog("Update Image Metadata")
def update_metadata(id, desc, names, dt, loc):
    _id = id
    print(_id)
    st.text_input(label="description:", value=desc)
    st.text_input(label="names", value=names)
    st.text_input(label="datetime", value=dt)
    st.text_input(label="location", value=loc)

    if st.button("Submit"):
        # st.session_state.vote = {"item": item, "reason": reason}
        st.rerun()


# iqn phase mono crystalline


def get_foldername_by_datetime():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def copy_folder_tree(src_path, dest_path):
    print(src_path, ":", dest_path)
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    shutil.copytree(src_path, dest_path, dirs_exist_ok=True)

# clean the specified path
def remove_files_folders(src_path):
    shutil.rmtree(src_path)

def replicate_folders_files(root_src_dir, root_dst_dir):
    directories_added = []
    files_added = []
    memory_used = []

    memory_used.append(shutil.disk_usage(root_dst_dir))
    for src_dir, dirs, files in os.walk(root_src_dir):
        print(src_dir, dirs, files)
        dst_dir = src_dir.replace(root_src_dir, root_dst_dir, 1)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
            directories_added.append(dst_dir)
        for file_ in files:
            src_file = os.path.join(src_dir, file_)
            dst_file = os.path.join(dst_dir, file_)
            if os.path.exists(dst_file):
                # in case of the src and dst are the same file
                # if os.path.samefile(src_file, dst_file):
                continue
            # os.remove(dst_file)
            shutil.copy(src_file, dst_dir)
            files_added.append(dst_file)

    memory_used.append(shutil.disk_usage(root_dst_dir))

    return (directories_added, files_added, memory_used)


"""
idx, state, total_memory, used, free
"""
def update_audit_records(audit_path, audit_file_name):
    if os.path.exists(os.path.join(audit_path, audit_file_name)):
        df = pd.read.csv(os.path.join(audit_path, audit_file_name))
        return df
    

# get  array of immediate child folders
def extract_user_raw_data_folders(path):
    return next(os.walk(path))[1]

# check if file exists at the path
def file_exists(fpath):
    return os.path.exists(fpath)
       
# file path to uuid string       
def create_uuid_from_string(val: str):
    hex_string = hashlib.md5(val.encode("UTF-8")).hexdigest()
    return str(uuid.UUID(hex=hex_string))       

def extract_subpath(base_path_str, full_path_str):
    base_path = Path(base_path_str)
    full_path = Path(full_path_str)
    try:
        subpath = full_path.relative_to(base_path)
        return str(subpath)
    except ValueError as e:
        st.error(f'Value Error occred {e} in extact_subpath')
        return None    
