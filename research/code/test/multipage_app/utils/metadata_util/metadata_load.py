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


def createVectorDB(vdp, icn, tcn):
    # vector database persistance
    client = cdb.PersistentClient(path=vdp, settings=Settings(allow_reset=True))

    # reset chromadb persistant store
    # client.reset()

    # list of collections
    collections_list = [c.name for c in client.list_collections()]

    # openclip embedding function!
    embedding_function = OpenCLIPEmbeddingFunction()

    # Image collection inside vector database 'chromadb'
    image_loader = ImageLoader()

    # init LLM modules
    # m, t, p = LLM.setLLM()

    # collection images defined
    collection_images = client.get_or_create_collection(
        name=icn, embedding_function=embedding_function, data_loader=image_loader
    )

    """
       Text collection inside vector database 'chromadb'
    """
    collection_text = client.get_or_create_collection(
        name=tcn,
        embedding_function=embedding_function,
    )


    return collection_images, collection_text


def config_load():
    with open("app_conf.yaml") as prop:
        dict = yaml.safe_load(prop)

        print(
            "* * * * * * * * * * * Metadata Generator Properties * * * * * * * * * * * *"
        )
        for k in dict.keys():
            print(f"{k} :  {dict[k]}  \n")

        print(
            "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *"
        )
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
