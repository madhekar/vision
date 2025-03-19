import json
import glob
import pandas as pd
#import util
import os
import uuid
import chardet
import chromadb as cdb
import streamlit as st

from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction as stef
from chromadb.utils.batch_utils import create_batches
from utils.util import model_util as mu
from utils.util import storage_stat as ss
from utils.config_util import config
from chromadb.utils.data_loaders import ImageLoader
from chromadb.config import Settings

"""
UPDATE METADATA STRUCTURE:
{
"ts": "1345295504.0", 
"names": "Esha", 
"uri": "/home/madhekar/work/home-media-app/data/input-data-1/img/AnjaliBackup/f623049e-bf89-5dcc-8628-3d310d6d4f96/vcm_s_kf_repr_832x624.jpg", 
"id": "dd90b7bd-9ae9-4c90-8e37-41c41cc5af69", 
"latlon": "(32.96887205555556, -117.18414305555557)", 
"loc": "13582, Sage Mesa Road, San Diego, San Diego County, California, 92130, United States", 
"text": "The image shows a modern building with a black exterior and glass windows. There is a sign on the front of the building that says \"CABS\" in white letters. 
There are four people in front of the building. Two of them are on the left side of the image, one is in the middle, and one is on the right side."
}

"""

def recur_listdir(path):
    print(f'----> {path}')
    try:
        for entry in os.listdir(path):
            f_path = os.path.join(path, entry)
            print(f'==>{path} :: {f_path}')
            if os.path.isdir(f_path):
                if f_path:
                   recur_listdir(f_path)
                else:
                    break   
            else:
                return f_path    
    except Exception as e:
        print(f'exception {e}')


def fileList(path, pattern='**/*', recursive=True):
    files = glob.glob(os.path.join(path, pattern), recursive=recursive)  
    return files      

def load_metadata(metadata_path, metadata_file, image_final_path, image_final_folder):
    data = []
    with open(os.path.join(metadata_path, metadata_file), mode="r") as f:
        for line in f:
            data.append(json.loads(line))

        df = pd.DataFrame(data)

        df["uri"] = df["uri"].str.replace(
            "input-data/img",
            "final-data/img/" + image_final_folder,
        )
        print(df.head(10))
    return df

def detect_encoding(fp):
    with open(fp, 'rb') as f:
        raw_data = f.read()
    res = chardet.detect(raw_data)
    return res['encoding']

def createVectorDB(df_data, vectordb_dir_path, image_collection_name, text_folder, text_collection_name):
    
    # vector database persistance
    client = cdb.PersistentClient( path=vectordb_dir_path, settings=Settings(allow_reset=True))

    # reset chromadb persistant store
    # client.reset()

    # list of collections
    collections_list = client.list_collections()
    #collections_list = [c.name for c in client.list_collections()]

    print(f'->>{collections_list}')
    # openclip embedding function!
    embedding_function = OpenCLIPEmbeddingFunction()

    # Image collection inside vector database 'chromadb'
    image_loader = ImageLoader()

    # collection images defined
    collection_images = client.get_or_create_collection(
        name=image_collection_name,
        embedding_function=embedding_function,
        data_loader=image_loader,
    )

    """
    IMAGE embeddings in vector database
    """
    if image_collection_name not in collections_list:
        # create list of image urls to embedded in vector db

        df_urls = df_data["uri"]

        # create unique uuids for each image
        df_ids = df_data["id"]

        df_metadatas = df_data[["ts", "latlon", "loc", "names", "text"]].T.to_dict().values()
  
        collection_images.add(ids=df_ids.tolist(), metadatas=list(df_metadatas), uris=df_urls.tolist())

        print(f"id: \n {df_ids.head()} \n metadata: \n {df_metadatas} \n url: \n {df_urls.head()} ")

    """
       Text collection inside vector database 'chromadb'
    """
    collection_text = client.get_or_create_collection(
        name=text_collection_name,
        embedding_function=embedding_function,
    )

    print(f'-->{text_folder}')
    """
      TEXT Embeddings on vector database
    """
    if text_collection_name not in collections_list:
        # text_pth = sorted(
        #     [
        #         document_name  # os.path.join(text_folder, document_name)
        #         for document_name in recur_listdir(text_folder)
        #         if document_name.endswith(".txt")
        #     ]
        # )

        text_pth = fileList(text_folder)
        print("=> text paths: \n", "\n".join(text_pth))

        list_of_text = []

        for text_f in text_pth:
            if os.path.isfile(text_f):
              try:  
                with open(text_f, encoding="ascii", errors='ignore') as f:
                    content = f.read()
                    list_of_text.append(content)   
              except FileNotFoundError:
                  st.error(f'file not found: {text_f}')          

        ids_txt_list = [str(uuid.uuid4()) for _ in range(len(list_of_text))]

        batches = create_batches(api=client,ids=ids_txt_list, documents=list_of_text)

        print("=> text generate ids:\n", len(ids_txt_list))
        print('-> ', len(list_of_text))
        for batch in batches:
            print(batch)
            collection_text.add(ids=batch[0], documents=batch[3])
     
        #collection_text.add(documents=list_of_text, ids=ids_txt_list)

    return collection_images, collection_text

'''
ok for now!
'''
def archive_metadata(metadata_path, arc_folder_name, metadata_file):
    new_archive_folder = os.path.join( metadata_path, arc_folder_name)
    if not os.path.exists(new_archive_folder):
        os.mkdir(new_archive_folder)
        os.rename(os.path.join(metadata_path, metadata_file), os.path.join(new_archive_folder, metadata_file))


def execute():
    (
        raw_data_path,
        image_initial_path,
        metadata_path,
        metadata_file,

        vectordb_path,
        image_collection_name,
        text_collection_name,
        video_collection_name,
        audio_collection_name,
        text_folder_name,

        image_final_path,
        text_final_path,
        video_final_path,
        audeo_final_path
    ) = config.vectordb_config_load()

    arc_folder_name = mu.get_foldername_by_datetime()

    st.sidebar.subheader("Storage Source", divider="gray")

    user_source_selected = st.sidebar.selectbox(
        "data source folder",
        options=ss.extract_user_raw_data_folders(raw_data_path),
        label_visibility="collapsed",
    )

    image_initial_path = os.path.join(image_initial_path, user_source_selected)
    image_final_path = os.path.join(image_final_path, user_source_selected)
    if not os.path.exists(image_final_path):
        os.makedirs(image_final_path)

    #copy images in input-data to final-data/datetime
    # mu.copy_folder_tree(image_initial_path, os.path.join(image_final_path, arc_folder_name) )

    metadata_path = os.path.join(metadata_path, user_source_selected)
    text_folder_name = os.path.join(text_folder_name, user_source_selected)

    b_load_metadata = st.button("load image metadata")
    if b_load_metadata:

        df_metadata = load_metadata(metadata_path=metadata_path, metadata_file=metadata_file, image_final_path=image_final_path, image_final_folder=arc_folder_name)

        createVectorDB(df_metadata, vectordb_path, image_collection_name, text_folder_name, text_collection_name)

        # archive_metadata(metadata_path, arc_folder_name, metadata_file)

        # mu.remove_files_folders(image_initial_path)
     

if __name__=='__main__':
    execute()

