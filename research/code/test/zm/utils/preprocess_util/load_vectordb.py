import json
import glob
import pandas as pd
import os
import uuid
import chardet
from datetime import datetime
import chromadb as cdb
import streamlit as st
import PIL
from PIL import ImageFile
from pathlib import Path
import textract as tex
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction as stef
from chromadb.utils.batch_utils import create_batches
from chromadb.config import DEFAULT_TENANT, Settings
from utils.util import model_util as mu
from utils.util import storage_stat as ss
from utils.util import video_util as vu
from utils.config_util import config
from chromadb.utils.data_loaders import ImageLoader
from chromadb.config import Settings
ImageFile.LOAD_TRUNCATED_IMAGES = True
from concurrent.futures import ThreadPoolExecutor

PIL.Image.MAX_IMAGE_PIXELS = 933120000
"""
UPDATE METADATA STRUCTURE:
{
"ts": "1345295504.0", 
"names": "Esha", 
"uri": "/home/madhekar/work/home-media-app/data/input-data/img/AnjaliBackup/f623049e-bf89-5dcc-8628-3d310d6d4f96/vcm_s_kf_repr_832x624.jpg", 
"id": "dd90b7bd-9ae9-4c90-8e37-41c41cc5af69", 
"latlon": "(32.96887205555556, -117.18414305555557)", 
"loc": "13582, Sage Mesa Road, San Diego, San Diego County, California, 92130, United States", 
"text": "The image shows a modern building with a black exterior and glass windows. There is a sign on the front of the building that says \"CABS\" in white letters. 
There are four people in front of the building. Two of them are on the left side of the image, one is in the middle, and one is on the right side."
}

{
"uri": "/home/madhekar/work/home-media-app/data/input-data/img/AnjaliBackup/19951e8d-2921-566b-b23c-dd08ddfb25de/vcm_s_kf_repr_960x540.jpg", 
"id": "20be6b77-a8f1-4b43-b6de-5f4194006e04", 
"ts": "1345132415.0", 
"latlon": "(32.9687, -117.184196)", 
"loc": "madhekar residence at carmel vally san diego, california", 
"names": "", 
"attrib": "angry", 
"text": "The image shows a group of people gathered in a large room. There are two people in the front row, both with their backs turned to the camera. 
They are wearing dark suits and ties, and look very serious. The other people in the room are wearing a variety of clothing, and are sitting in rows facing the front row.
\n\nThe room is decorated with a large painting on the wall behind the front row. The painting is of a group of people gathered in a field, with a mountain range in the background. 
The people in the painting are all wearing casual clothing, and look happy and relaxed.\n\nIn the background, there is a large window looking out at a city. 
The city is in the distance, and is covered in a haze. The people in the painting are looking out at the city as well.\n\nIn the foreground, there is a table with a few chairs around it. 
The table is covered in a white tablecloth, and there is a v"
}

import chromadb
from chromadb.utils.batch_utils import create_batches
import uuid

# 1. Initialize the Chroma client
# Use PersistentClient for data to survive program restarts
client = chromadb.PersistentClient(path="test-large-batch")

# 2. Create a large dummy dataset (e.g., 100,000 items)
# Each item is a tuple: (id, document, embedding)
large_batch_data = [(f"{uuid.uuid4()}", f"document {i}", [0.1] * 1536) for i in range(100000)]

# Unpack the data into separate lists for the `create_batches` function
ids, documents, embeddings = zip(*large_batch_data)
ids_list, documents_list, embeddings_list = list(ids), list(documents), list(embeddings)

# 3. Use create_batches to split the data
# The function automatically determines the appropriate batch size based on the client's limits
batches = create_batches(
    api=client,
    ids=ids_list,
    documents=documents_list,
    embeddings=embeddings_list
)

# 4. Get or create a collection
collection = client.get_or_create_collection("test_collection")

# 5. Iterate through the batches and add them to the collection
for batch in batches:
    # Each 'batch' is a tuple: (ids, embeddings, metadatas, documents)
    # The order for 'collection.add' is (ids, embeddings, metadatas, documents)
    print(f"Adding batch of size {len(batch[0])}")
    collection.add(
        ids=batch[0],
        embeddings=batch[1],
        metadatas=batch[2],
        documents=batch[3]
    )

print("Finished adding all data in batches.")


-----

The most straight forward approach would probably 
be to combine the int representation of two UUIDs with bitwise operators and construct a new UUID from it:

>>> from uuid import *

>>> u1 = uuid4()
>>> u2 = uuid4()
>>> u3 = UUID(int=u1.int ^ u2.int, version=4)
>>> u1, u2, u3
(UUID('2266aff1-a7be-4c71-bc0d-987779f68bd3'),
 UUID('284c5065-299f-479c-9d6b-d353012795d7'), 
 UUID('0a2aff94-8e21-4bed-a166-4b2478d11e04'))

I can't tell you what the best operator for combining here would be, an XOR, OR, AND, or whatever else. 

To combine two UUIDs into a single valid UUIDv4, you must merge their data while preserving the specific bits that define it as a "Version 4" identifier. 
Directly concatenating them or using raw bitwise operations without correction will result in a value that is no longer a valid UUID. 
Recommended Methods for Combining UUIDs

    Bitwise XOR (Python Approach): The most efficient way is to combine the integer representations using the XOR operator. 
    To ensure the result remains a valid UUIDv4, you must manually set the version (4) and variant (RFC 4122) bits.
    python

    import uuid
    u1 = uuid.UUID('...')
    u2 = uuid.UUID('...')
    # Combine bits and force version 4
    combined = uuid.UUID(int=u1.int ^ u2.int, version=4)

    Hashing (UUIDv5 Approach): If you need the result to be deterministic (the same two inputs always produce the same output), use a name-based UUID like UUIDv5. 
    This hashes a "namespace" UUID with a "name" string (the second UUID) to create a new one.
    python

    import uuid
    u1 = uuid.UUID('...') # Acts as namespace
    u2_str = '...'         # Acts as name
    combined = uuid.uuid5(u1, u2_str)

    Concatenation (Non-UUID Result): If the final ID does not have to be a 128-bit UUID, simply concatenating the strings with a delimiter 
    (e.g., uuid1:uuid2) is the only way to avoid data loss, as you cannot fit 256 bits of entropy into a 128-bit slot without the risk of collisions. 

Critical Considerations

    Validity: A standard UUIDv4 must have the digit 4 at the 13th character position and one of 8, 9, a, b at the 17th position. 
    Using a library's version=4 parameter (like in the Python uuid module) handles this automatically.
    Collision Risk: Combining two UUIDs does not mathematically decrease the chance of a collision; 
    it remains dependent on the 122 bits of entropy available in the final UUIDv4 structure.
    Information Loss: Because two 128-bit values are being compressed into one 128-bit value, the process is lossy. 
    You cannot "un-combine" the result to get the original two UUIDs back unless you store the mapping separately. 

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
        df['uri'] =  df.apply(vu.extract_video_paths_from_metadata, axis=1)
        df_e = df.explode(['uri'])
               
        # create id for each frame       
        df_e['id'] = df_e['uri'].apply(mu.create_uuid_from_string)
        df_e.reset_index(drop=True, inplace=True)
        
        df_e["uri"] = df_e["uri"].str.replace(
         "input-data/video",
         "final-data/video" #+ image_final_path,
        )
        print(df_e.head(20))
    return df_e

# handle new creation on metadata file from scratch
def load_metadata(metadata_path, metadata_file, image_final_path, image_final_folder):
    data = []
    with open(os.path.join(metadata_path, metadata_file), mode="r") as f:
        for line in f:
            data.append(json.loads(line))

        df = pd.DataFrame(data)

        df["uri"] = df["uri"].str.replace(
            "input-data/img",
            "final-data/img" #+ image_final_path,
        )
        print(df.head(10))
    return df

def detect_encoding(fp):
    with open(fp, 'rb') as f:
        raw_data = f.read()
    res = chardet.detect(raw_data)
    return res['encoding']

# Function to add to collection
def add_imgs_to_vector_db(collection, ids, uris, metadatas):
    collection.add(
        ids=ids,
        uris=uris,
        metadatas=metadatas
    )
    print(f"Added {len(ids)}")

def createVectorDB(df_data, df_video_data, vector_db_dir_path, image_collection_name, text_folder, text_collection_name, video_collection_name, max_workers=20):
    
    cdb.api.client.SharedSystemClient.clear_system_cache()
    # vector database persistence
    client = cdb.PersistentClient( path=vector_db_dir_path, tenant=DEFAULT_TENANT, settings=Settings(allow_reset=True))

    client.clear_system_cache()

    # list of collections
    collections_list = client.list_collections()

    st.info(f'Info: Existing collections: {collections_list}')

    if len(collections_list) == 0:    

        # openclip embedding function!
        embedding_function = OpenCLIPEmbeddingFunction()

        #text_embedding_function = OpenCLIPEmbeddingFunction(model_name="coca_roberta-ViT-B-32")

        """ 
        Image collection inside vector database 'chromadb'
        """
        image_loader = ImageLoader()

        """
        # Images collection defined
        """
        collection_images = client.get_or_create_collection(
            name=image_collection_name,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"},
            data_loader=image_loader,
        )    

        """ 
        Videos collectoion defined
        """
        collection_videos = client.get_or_create_collection(
           name=video_collection_name,
           embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"},
            data_loader=image_loader,
        )

        """
          Text collection inside vector database 'chromadb'
        """
        collection_text = client.get_or_create_collection(
            name=text_collection_name,
            #metadata={"hnsw:space": "cosine"},
            embedding_function=embedding_function,
        )
    else:
        for collection in collections_list:
          client.delete_collection(collection)

        # reset chromadb persistant store
        #client.reset()

        #openclip embedding function!
        embedding_function = OpenCLIPEmbeddingFunction()

        # Image collection inside vector database 'chromadb'
        image_loader = ImageLoader()

        """
        # collection images defined
        # """
        collection_images = client.get_or_create_collection(
            name=image_collection_name,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"},
            data_loader=image_loader,
        )

        """
        # collection images defined
        """
        collection_videos = client.get_or_create_collection(
           name=video_collection_name,
           embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"},
            data_loader=image_loader,
        )
        
        """
          Text collection inside vector database 'chromadb'
        """
        collection_text = client.get_or_create_collection(
            name=text_collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=embedding_function,
        )

    """
    IMAGE embeddings in vector database
    """
       
    df_uris =  df_data['uri']
    df_ids = df_data['id']
    df_metadata = df_data[["ts", "src", "type", "latlon", "loc", "ppt", "caption", "text"]].fillna("").T.to_dict().values()

    collection_images.add(ids=df_ids.tolist(), metadatas=list(df_metadata), uris=df_uris.tolist()) 

    
    # ls_metadatas = df_data[["ts","type", "latlon", "loc", "ppt", "text"]].T.to_dict().values()

    # with ThreadPoolExecutor(max_workers) as executor:
    #     executor.map(add_imgs_to_vector_db, df_data["id"].tolist(), df_data["uri"].tolist(), ls_metadatas)
    
    st.info(f"Info: Done adding number of images: {len(df_uris)}")

    """
    VIDEO embedding in vector database
    uri, id, ts, latlon, loc, text
    """
    print("----->>", df_video_data.head())
    df_video_uris = df_video_data['uri']  # frame uri
    df_video_ids = df_video_data['id']  # frame id
    df_video_metadata = df_video_data[["ts", "latlon", "loc", "text", "vuri"]].fillna("").T.to_dict().values()

    collection_videos.add(ids=df_video_ids.tolist(), uris=df_video_uris.tolist(), metadatas=list(df_video_metadata))

    st.info(f"Info: Done adding number of frames for videos: {len(df_video_uris)}")

    """
      TEXT Embeddings on vector database
    """
    #if text_collection_name not in collections_list:

    text_pth = fileList(text_folder)
    if len(text_pth) > 0:
        #st.info("text paths: \n", "\n".join(text_pth))

        list_of_text = []
        meta = []
        for text_f in text_pth:
            if os.path.isfile(text_f):
                
                print(f"File Name: {text_f}")
                try:  
                    val = tex.process(text_f)
                    #with open(text_f, 'r', encoding="utf-8", errors='replace') as f:
                    content = val.decode("utf-8")
                    print(f"======>> {content}")
                    list_of_text.append(content)   
                    meta.append({"name": text_f, "ts": str(datetime.now())})
                except UnicodeDecodeError as e:
                    st.error(f'error: ignoring the text file, could not decode file as ascii: {e}')      
                except FileNotFoundError:
                    st.error(f'error: text file not found: {text_f}')          

        ids_txt_list = [str(uuid.uuid4()) for _ in range(len(list_of_text))]

        batches = create_batches(api=client, ids=ids_txt_list, metadatas= meta, documents=list_of_text)

        st.info(f"number of ids: {len(ids_txt_list)}")
        st.info(f'number of documents: {len(list_of_text)}')
        st.info(f"starting to add documents in number of batches: {len(batches)}")

        for batch in batches:
            print(f"---batch-->: {batch}")
            collection_text.add(ids=batch[0],
                                embeddings=batch[1],
                                metadatas=batch[2],
                                documents=batch[3])

        st.info(f"done adding documents: {len(list_of_text)}")
    else:
        st.warning(f'no text documents found in {text_folder}')    

    client.clear_system_cache()

    return collection_images, collection_text

'''
ok for now! todo
'''
def archive_metadata(metadata_path, arc_folder_name, metadata_file):
    new_archive_folder = os.path.join( metadata_path) #, arc_folder_name)
    if not os.path.exists(new_archive_folder):
        os.mkdir(new_archive_folder)
        os.rename(os.path.join(metadata_path, metadata_file), os.path.join(new_archive_folder, metadata_file))

def final_multimedia_path(f_path, user_selection):
    f_path = os.path.join(f_path, user_selection)
    if not os.path.exists(f_path):
        os.makedirs(f_path)
    else:
        print(f'---->{f_path}')
        mu.remove_files_folders(f_path)   
        os.makedirs(f_path)
    return f_path    

def execute():
    (
        raw_data_path,
        image_initial_path,
        video_initial_path,
        metadata_path,
        metadata_file,
        video_metadata_file,
        max_workers,

        vectordb_path,
        image_collection_name,
        text_collection_name,
        video_collection_name,
        audio_collection_name,
        text_folder_name,

        image_final_path,
        text_final_path,
        video_final_path,
        audio_final_path
    ) = config.vectordb_config_load()

    arc_folder_name = mu.get_foldername_by_datetime()

    st.sidebar.subheader("Storage Source", divider="gray")
    user_source_selected = st.sidebar.selectbox(
        "data source folder",
        options=ss.extract_user_raw_data_folders(raw_data_path),
        label_visibility="collapsed",
    )

    image_initial_path = os.path.join(image_initial_path, user_source_selected)
    video_initial_path = os.path.join(video_initial_path, user_source_selected)

    image_final_path = final_multimedia_path(image_final_path, user_source_selected)
    video_final_path = final_multimedia_path(video_final_path, user_source_selected)
    text_final_path = final_multimedia_path(text_final_path, user_source_selected)

    print(f'final paths: {image_final_path} : {video_final_path} : { text_final_path}')

    #copy images in input-data to final-data/datetime
    mu.copy_folder_tree(image_initial_path, image_final_path)

    #copy videos and frames from input-data to final-data
    mu.copy_folder_tree(video_initial_path, video_final_path)

    metadata_path = os.path.join(metadata_path, user_source_selected)
    text_folder_name = os.path.join(text_folder_name, user_source_selected)

    b_load_metadata = st.button("load image metadata", type="primary")
    if b_load_metadata:

        df_metadata = load_metadata(metadata_path=metadata_path, metadata_file=metadata_file, image_final_path=image_final_path, image_final_folder=arc_folder_name)

        df_video_metadata = load_video_metadata(metadata_path=metadata_path, metadata_file=video_metadata_file, image_final_path=image_final_path, image_final_folder=arc_folder_name)

        createVectorDB(df_metadata, df_video_metadata, vectordb_path, image_collection_name, text_folder_name, text_collection_name, video_collection_name, max_workers)

        archive_metadata(metadata_path, arc_folder_name, metadata_file)

        #mu.remove_files_folders(image_initial_path)
        #mu.remove_files_folders(text_folder_name)

if __name__=='__main__':
    execute()

