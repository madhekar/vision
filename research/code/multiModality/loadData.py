import os
import uuid
import streamlit as st
from dotenv import load_dotenv

import chromadb as cdb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from transformers import AutoTokenizer, AutoProcessor, AutoModel, TextStreamer
from chromadb.utils.data_loaders import ImageLoader
from chromadb.config import Settings

# vector database path
load_dotenv('/home/madhekar/.env.local')
storage_path = os.getenv('STORAGE_PATH')

if storage_path is None:
    raise ValueError("STORAGE_PATH environment variable is not set")

# Get the uris to the images
IMAGE_FOLDER = '/home/madhekar/work/vision/research/code/image_embed/flowers/allflowers'


def createVectorDB():

    # vector database persistance
    client = cdb.PersistentClient( path=storage_path, settings=Settings(allow_reset=True))

    # reset chromadb persistant store
    client.reset()

    # openclip embedding function!
    embedding_function = OpenCLIPEmbeddingFunction()

    # Image collection inside vector database 'chromadb'
    image_loader = ImageLoader()

    # collection images define
    collection_images = client.get_or_create_collection(
      name='multimodal_collection_images', 
      embedding_function=embedding_function, 
      data_loader=image_loader
      )

    # add image embeddings in vector db
    image_uris = sorted([os.path.join(IMAGE_FOLDER, image_name) for image_name in os.listdir(IMAGE_FOLDER) if not image_name.endswith('.txt')])
    ids = [str(uuid.uuid4()) for _ in range(len(image_uris))]
    
    print('=> image urls: \n', '\n'.join(image_uris))
    collection_images.add(ids=ids, uris=image_uris)

    '''
       Text collection inside vector database 'chromadb'
    '''
    collection_text = client.get_or_create_collection(
      name="multimodal_collection_text",
      embedding_function=embedding_function,
    )

    text_pth = sorted([
        os.path.join(IMAGE_FOLDER, image_name)
        for image_name in os.listdir(IMAGE_FOLDER)
        if image_name.endswith(".txt")
    ])

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

@st.cache_resource
def setLLM():
    '''
        model autotokenizer and processor componnents for LLM model MC-LLaVA-3b with trust flag
    '''

    model = AutoModel.from_pretrained("visheratin/MC-LLaVA-3b", trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained( "visheratin/MC-LLaVA-3b", trust_remote_code=True)

    processor = AutoProcessor.from_pretrained( "visheratin/MC-LLaVA-3b", trust_remote_code=True)

    return model, tokenizer, processor


@st.cache_resource
def init():
    return createVectorDB()
