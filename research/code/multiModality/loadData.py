import os

import uuid
import streamlit as st
from dotenv import load_dotenv
import util

import chromadb as cdb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

from chromadb.utils.data_loaders import ImageLoader
from chromadb.config import Settings

# vector database path
load_dotenv('/home/madhekar/.env.local')
storage_path = os.getenv('STORAGE_PATH')

if storage_path is None:
    raise ValueError("STORAGE_PATH environment variable is not set")

# Get the uris to the images
IMAGE_FOLDER = '/home/madhekar/work/zsource/family/img'
DOCUMENT_FOLDER = "/home/madhekar/work/zsource/family/doc"

def createVectorDB():

    # vector database persistance
    client = cdb.PersistentClient( path=storage_path, settings=Settings(allow_reset=True))

    # reset chromadb persistant store
    # client.reset()
    
    #list of collections
    collections_list = [c.name for c in client.list_collections()]

    # openclip embedding function!
    embedding_function = OpenCLIPEmbeddingFunction()

    # Image collection inside vector database 'chromadb'
    image_loader = ImageLoader()

    # collection images defined
    collection_images = client.get_or_create_collection(
      name='multimodal_collection_images', 
      embedding_function=embedding_function, 
      data_loader=image_loader
      )
    if 'multimodal_collection_images' not in collections_list:
       # add image embeddings in vector db
       image_uris = sorted([os.path.join(IMAGE_FOLDER, image_name) for image_name in os.listdir(IMAGE_FOLDER) if not image_name.endswith('.txt')])
    
       #create uuids for each image
       ids = [str(uuid.uuid4()) for _ in range(len(image_uris))]
    
       # create metadata for each image
       metadata = []
       for url in image_uris:
          v = util.getMetadata(url)
          metadata.append({'year': v[0], 'month':v[1], 'day': v[2], 'datetime': v[3], 'location': v[4]})
    
    
       print('=> image urls: \n', '\n'.join(image_uris))
       collection_images.add(ids=ids, metadatas=metadata, uris=image_uris)

    '''
       Text collection inside vector database 'chromadb'
    '''
    collection_text = client.get_or_create_collection(
      name="multimodal_collection_text",
      embedding_function=embedding_function,
    )
    
    if 'multimodal_collection_text' not in collections_list:
      text_pth = sorted(
        [
            os.path.join(DOCUMENT_FOLDER, document_name)
            for document_name in os.listdir(DOCUMENT_FOLDER)
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


@st.cache_resource(show_spinner=True)
def init():
    return createVectorDB()
