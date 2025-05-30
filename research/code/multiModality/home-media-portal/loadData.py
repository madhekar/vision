import os

import uuid
import streamlit as st
from dotenv import load_dotenv
import util
from entities import getEntityNames
import LLM
import zasync as zas
import yaml

import chromadb as cdb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

from chromadb.utils.data_loaders import ImageLoader
from chromadb.config import Settings

# vector database path
#load_dotenv('/home/madhekar/.env.local')
#storage_path = os.getenv('STORAGE_PATH')

#if storage_path is None:
#    raise ValueError("STORAGE_PATH environment variable is not set")

# Get the uris to the images
#IMAGE_FOLDER = '/home/madhekar/work/zsource/family/img'
#IMAGE_FOLDER = '/home/madhekar/Pictures'
#DOCUMENT_FOLDER = "/home/madhekar/work/zsource/family/doc"
#image_collection = "multimodal_collection_images"

def createVectorDB(vdp, icn, tcn):

    # vector database persistance
    client = cdb.PersistentClient( path=vdp, settings=Settings(allow_reset=True))

    # reset chromadb persistant store
    #client.reset()
    
    #list of collections
    collections_list = [c.name for c in client.list_collections()]

    # openclip embedding function!
    embedding_function = OpenCLIPEmbeddingFunction()

    # Image collection inside vector database 'chromadb'
    image_loader = ImageLoader()

    # init LLM modules
    #m, t, p = LLM.setLLM()

    # collection images defined
    collection_images = client.get_or_create_collection(
      name=icn, 
      embedding_function=embedding_function, 
      data_loader=image_loader
      )
    
    '''
    IMAGE embeddings in vector database
    '''
    # if 'multimodal_collection_images' not in collections_list:
        
    #     # create list of image urls to embedded in vector db
    #     image_uris = sorted(util.getRecursive(IMAGE_FOLDER))

    #     # create unique uuids for each image
    #     ids = [str(uuid.uuid4()) for _ in range(len(image_uris))]

    #     # create metadata for each image
    #     metadata = []
    #     for url in image_uris:
    #         """
    #            extract metadata for the url such as date, time, location
    #         """
    #         v = util.getMetadata(url)
    #         """
    #           extract image caption from the model
    #         """
    #         n = getEntityNames(url)
    #         """
    #           get LLM description of image from the model.
    #         """
    #         d = LLM.fetch_llm_text(
    #             url,
    #             model=m,
    #             processor=p,
    #             top=0.9,
    #             temperature=0.9,
    #             question="Answer with organized thoughts: Please describe the picture, ",
    #             people=n,
    #             location=v[3]
    #         )

    #         metadata.append(
    #             {
    #                 "ts": v[0],
    #                 "lat": v[1],
    #                 "lon": v[2],
    #                 "loc": v[3],
    #                 "nam": str(n),
    #                 "txt": str(d),
    #             }
    #         )
    #         st.write('metadata: ', v, ' : ', n, ' : ', d)

    #     #print("=> image urls: \n", "\n".join(image_uris))
    #     collection_images.add(ids=ids, metadatas=metadata, uris=image_uris)

    '''
       Text collection inside vector database 'chromadb'
    '''
    collection_text = client.get_or_create_collection(
      name=tcn,
      embedding_function=embedding_function,
    )

    '''
      TEXT Embeddings on vector database
    '''
    # if 'multimodal_collection_text' not in collections_list:
    #     text_pth = sorted(
    #         [
    #             os.path.join(DOCUMENT_FOLDER, document_name)
    #             for document_name in os.listdir(DOCUMENT_FOLDER)
    #             if document_name.endswith(".txt")
    #         ]
    #     )

    #     print("=> text paths: \n", "\n".join(text_pth))

    #     list_of_text = []

    #     for text in text_pth:
    #         with open(text, "r") as f:
    #             text = f.read()
    #             list_of_text.append(text)

    #     ids_txt_list = [
    #         str(uuid.uuid4())
    #         for _ in range(len(list_of_text))
    #     ]

    #     print("=> text generate ids:\n", ids_txt_list)

    #     collection_text.add(documents=list_of_text, ids=ids_txt_list)

    return collection_images, collection_text

def config_load():
    with open("app_conf.yaml") as prop:
        dict = yaml.safe_load(prop)

        print("* * * * * * * * * * * Metadata Generator Properties * * * * * * * * * * * *")
        for k in dict.keys():
            print(f"{k} :  {dict[k]}  \n")

        print("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *")
        vectordb_dir_path = dict["vectordb"]["vectordb_path"] 
        image_collection_name = dict["vectordb"]["image_collection_name"]
        text_collection_name = dict["vectordb"]["text_collection_name"]

    return (vectordb_dir_path, image_collection_name, text_collection_name)

@st.cache_resource(show_spinner=True)
def init():
    (vdp, icn, tcn) = config_load()
    return createVectorDB(vdp, icn, tcn)

# def updateMetadata(id, desc, names, dt, loc):
#     # vector database persistance
#     client = cdb.PersistentClient(path=storage_path, settings=Settings(allow_reset=True))
#     col = client.get_collection(image_collection)
#     col.update(
#         ids=id,
#         metadatas={"description": desc, "names": names, "datetime" : dt, "location": loc}
#     )