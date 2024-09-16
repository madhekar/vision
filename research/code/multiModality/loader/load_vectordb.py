import json
import pandas as pd
import yaml
import util
import os
import uuid
import chromadb as cdb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

from chromadb.utils.data_loaders import ImageLoader
from chromadb.config import Settings

def load_metadata(metadata_path, metadata_file):
  data = []
  with open(os.path.join(metadata_path, metadata_file), mode="r") as f:
    for line in f:
        data.append(json.loads(line))

    df =  pd.DataFrame(data)
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
  
        collection_images.add(ids=df_ids.tolist(), metadatas=df_metadatas, uris=df_urls.tolist())

        print(f"id: \n {df_ids.head()} \n metadata: \n {df_metadata} \n url: \n {df_urls.head()} ")

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
     

if __name__=='__main__':
    with open("metadata_conf.yaml") as prop:
        dict = yaml.safe_load(prop)

        print("* * * * * * * * * * * * Metadata Load Properties * * * * * * * * * * * * * *")
        print(dict)
        print("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *")

    metadata_path = dict["metadata"]["metadata_path"]
    metadata_file = dict["metadata"]["metadata_file"]
    vectordb_dir_path = dict["vectordb"]["vectordb_path"]   
    image_collection_name = dict["vectordb"]["image_collection_name"]
    text_collection_name = dict["vectordb"]["text_collection_name"]
    text_folder_name = dict["vectordb"]["text_dir_path"]

    arc_folder_name = util.get_foldername_by_datetime()

    df_metadata = load_metadata(metadata_path=metadata_path, metadata_file=metadata_file)

    createVectorDB(df_metadata, vectordb_dir_path, image_collection_name, text_folder_name, text_collection_name)

    archive_metadata(metadata_path, arc_folder_name, metadata_file)