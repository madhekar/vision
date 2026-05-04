import os
import json
import pandas as pd
import hashlib
import uuid
from pathlib import Path
'''
Yes, using a separate ChromaDB collection per modality (e.g., text, images, audio) within a PersistentClient is generally preferred and recommended. 
This approach provides better control over embedding models, search accuracy, and performance optimization.Here is why this structure is beneficial, 
along with best practices:Why Separate Collections Per Modality?Different Embedding Models: Each modality typically requires a unique embedding function
(e.g., CLIP for images vs. SentenceTransformers for text). A single Chroma collection requires a single, consistent embedding model configuration.
Performance Optimization: You can tailor indexing options (e.g., distance functions) to the specific data type.
Easier Query Management: Querying only images is faster and more accurate if they are separated from the text corpus.When to Use One Collection 
(Multi-modal)If you are using a unified multi-modal embedding model (e.g., OpenCLIP) that maps both images and text into the exact same vector space, 
you can use a single collection. However, this is less common than separating them.Best Practices for PersistentClientUse One Client: 
Create one PersistentClient instance and reuse it in your project to avoid data access issues.Set the Path: Use chromadb.PersistentClient(path="./path_to_data") 
to ensure your data is saved to disk rather than stored in-memory, preventing loss on exit.get_or_create_collection: Use this method to securely load or create your collections.
Example Structure:pythonimport chromadb

# Initialize persistent client
client = chromadb.PersistentClient(path="./my_data")

# Create separate collections for different modalities
text_collection = client.get_or_create_collection(name="text_data")
image_collection = client.get_or_create_collection(name="image_data")
Use code with caution.
'''
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