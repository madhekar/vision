import os
import json
import pandas as pd
import hashlib
import uuid
from pathlib import Path

# file path to uuid string       
def create_uuid_from_string(val: str):
    hex_string = hashlib.md5(str(val).encode("UTF-8")).hexdigest()
    return str(uuid.UUID(hex=hex_string))  

def extract_video_paths_from_metadata(row):
    vpath = Path(row['vuri'])
    root_path = os.path.split(vpath)[0]
    parent = vpath.stem
    frames = "frames"
    
    img_frame_list = []
    frames_path = os.path.join(root_path, frames, parent)
    #print(frames_path)
    if os.path.exists(frames_path):
        img_frame_list = [os.path.join(frames_path, ifile) for ifile in  os.listdir(frames_path)]
    return img_frame_list

# def gen_uuid(row):
#     return [create_uuid_from_string(u) for u in row['uri']]

# handle new creation on metadata file from scratch
def load_video_metadata(metadata_path, metadata_file, image_final_path, image_final_folder):
    data = []
    with open(os.path.join(metadata_path, metadata_file), mode="r") as f:
        for line in f:
            data.append(json.loads(line))

        # clean video ids and uri
        df = pd.DataFrame(data)
        df.rename(columns={"uri": "vuri"}, inplace=True)
        df = df.drop(columns=['id']) 

        # create uri for each frame
        df['uri'] =  df.apply(extract_video_paths_from_metadata, axis=1)
        df_e = df.explode(['uri'])
               
        # create id for each frame       
        df_e['id'] = df_e['uri'].apply(create_uuid_from_string)
        df_e.reset_index(drop=True, inplace=True)
        
        # df["uri"] = df["uri"].str.replace(
        #     "input-data/img",
        #     "final-data/img" #+ image_final_path,
        # )
        print(df_e.head(20))
    return df


vmpath ="/mnt/zmdata/home-media-app/data/app-data/metadata/Berkeley/"
vmfile ="video_metadata.json"

load_video_metadata(vmpath, vmfile, "", "")