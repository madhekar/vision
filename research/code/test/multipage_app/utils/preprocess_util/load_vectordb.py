import json
import pandas as pd
#import util
import os
import uuid
import chromadb as cdb
import streamlit as st

from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
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


def load_metadata(metadata_path, metadata_file, image_final_path, image_final_folder):
    data = []
    with open(os.path.join(metadata_path, metadata_file), mode="r") as f:
        for line in f:
            data.append(json.loads(line))

        df = pd.DataFrame(data)

        df["url"] = df["url"].str.replace(
            "input-data/img",
            "final-data/img/" + image_final_folder,
        )
    return df


def createVectorDB(df_data, vectordb_dir_path, image_collection_name, text_folder, text_collection_name):
    
    # vector database persistance
    client = cdb.PersistentClient( path=vectordb_dir_path, settings=Settings(allow_reset=True))

    # reset chromadb persistant store
    # client.reset()

    # list of collections
    collections_list = [c.name for c in client.list_collections()]

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

        df_urls = df_data["url"]

        # create unique uuids for each image
        df_ids = df_data["id"]

        df_metadatas = df_data[["timestamp", "lat", "lon", "loc", "nam", "txt"]].T.to_dict().values()
  
        collection_images.add(ids=df_ids.tolist(), metadatas=list(df_metadatas), uris=df_urls.tolist())

        print(f"id: \n {df_ids.head()} \n metadata: \n {df_metadatas} \n url: \n {df_urls.head()} ")

    """
       Text collection inside vector database 'chromadb'
    """
    collection_text = client.get_or_create_collection(
        name=text_collection_name,
        embedding_function=embedding_function,
    )

    """
      TEXT Embeddings on vector database
    """
    if text_collection_name not in collections_list:
        text_pth = sorted(
            [
                os.path.join(text_folder, document_name)
                for document_name in os.listdir(text_folder)
                if document_name.endswith(".txt")
            ]
        )

        print("=> text paths: \n", "\n".join(text_pth))

        list_of_text = []

        for text in text_pth:
            with open(text, "r") as f:
                text = f.read()
                list_of_text.append(text)

        ids_txt_list = [str(uuid.uuid4()) for _ in range(len(list_of_text))]

        print("=> text generate ids:\n", ids_txt_list)

        collection_text.add(documents=list_of_text, ids=ids_txt_list)

    return collection_images, collection_text


def archive_metadata(metadata_path, arc_folder_name, metadata_file):
    new_archive_folder = os.path.join(
        metadata_path,
        arc_folder_name,
    )
    if not os.path.exists(new_archive_folder):
        os.mkdir(new_archive_folder)
        os.rename(os.path.join(metadata_path, metadata_file), os.path.join(new_archive_folder, metadata_file))


def execute():
    (
        image_initial_path,
        metadata_path,
        metadata_file,

        vectordb_path,
        image_collection_name,
        text_collection_name,
        video_collection_name,
        text_folder_name,

        image_final_path,
        text_final_path,
        video_final_path
    ) = config.vectordb_config_load()

    arc_folder_name = mu.get_foldername_by_datetime()

    st.sidebar.subheader("Storage Source", divider="gray")

    user_source_selected = st.sidebar.selectbox(
        "data source folder",
        options=ss.extract_user_raw_data_folders(image_initial_path),
        label_visibility="collapsed",
    )

    image_initial_path = os.path.join(image_initial_path, user_source_selected)

    #copy images in input-data to final-data/datetime
    mu.copy_folder_tree(image_initial_path, os.path.join(image_final_path, arc_folder_name) )

    df_metadata = load_metadata(metadata_path=metadata_path, metadata_file=metadata_file, image_final_path=image_final_path, image_final_folder=arc_folder_name)

    createVectorDB(df_metadata, vectordb_path, image_collection_name, text_folder_name, text_collection_name)

    archive_metadata(metadata_path, arc_folder_name, metadata_file)

    mu.remove_files_folders(image_initial_path)
     

if __name__=='__main__':
    execute()

