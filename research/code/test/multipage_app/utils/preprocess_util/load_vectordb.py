import json
import glob
import pandas as pd
import os
import uuid
import chardet
import chromadb as cdb
import streamlit as st

from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction as stef
from chromadb.utils.batch_utils import create_batches
from chromadb.config import DEFAULT_TENANT, Settings
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

{
"uri": "/home/madhekar/work/home-media-app/data/input-data-1/img/AnjaliBackup/19951e8d-2921-566b-b23c-dd08ddfb25de/vcm_s_kf_repr_960x540.jpg", 
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
    client = cdb.PersistentClient( path=vectordb_dir_path, tenant=DEFAULT_TENANT, settings=Settings(allow_reset=True))

    client.clear_system_cache()

    # list of collections
    collections_list = client.list_collections()

    st.info(f'Info: Existing collections: {collections_list}')

    if len(collections_list) == 0:    

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
          Text collection inside vector database 'chromadb'
        """
        collection_text = client.get_or_create_collection(
            name=text_collection_name,
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

        # collection images defined
        collection_images = client.get_or_create_collection(
            name=image_collection_name,
            embedding_function=embedding_function,
            data_loader=image_loader,
        )

        """
          Text collection inside vector database 'chromadb'
        """
        collection_text = client.get_or_create_collection(
            name=text_collection_name,
            embedding_function=embedding_function,
        )

    """
    IMAGE embeddings in vector database
    """
    #if image_collection_name not in collections_list:
    # create list of image urls to embedded in vector db

    df_urls = df_data["uri"]

    # create unique uuids for each image
    df_ids = df_data["id"]

    df_metadatas = df_data[["ts", "latlon", "loc", "names", "text"]].T.to_dict().values()

    collection_images.add(ids=df_ids.tolist(), metadatas=list(df_metadatas), uris=df_urls.tolist())

    #st.info(f"id: \n {df_ids.head()} \n metadata: \n {df_metadatas} \n url: \n {df_urls.head()} ")

    st.info(f"Info: Done adding number of images: {len(df_urls)}")


    """
      TEXT Embeddings on vector database
    """
    #if text_collection_name not in collections_list:

    text_pth = fileList(text_folder)
    #st.info("text paths: \n", "\n".join(text_pth))

    list_of_text = []

    for text_f in text_pth:
        if os.path.isfile(text_f):
            try:  
              with open(text_f, 'r', encoding="ascii") as f:
                content = f.read()
                list_of_text.append(content)   
            except UnicodeDecodeError as e:
                st.error(f'error: ignoring the text file, could not decode file as ascii: {e}')      
            except FileNotFoundError:
                st.error(f'error: text file not found: {text_f}')          

    ids_txt_list = [str(uuid.uuid4()) for _ in range(len(list_of_text))]

    batches = create_batches(api=client,ids=ids_txt_list, documents=list_of_text)

    st.info(f"number of ids: {len(ids_txt_list)}")
    st.info(f'number of documents: {len(list_of_text)}')
    st.info(f"starting to add documents in number of batches: {len(batches)}")

    for batch in batches:
        collection_text.add(ids=batch[0], documents=batch[3])

    st.info(f"done adding documents: {len(list_of_text)}")

    client.clear_system_cache()

    return collection_images, collection_text

'''
ok for now!
'''
def archive_metadata(metadata_path, arc_folder_name, metadata_file):
    new_archive_folder = os.path.join( metadata_path, arc_folder_name)
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

    image_final_path = final_multimedia_path(image_final_path, user_source_selected)
    text_final_path = final_multimedia_path(text_final_path, user_source_selected)

    print(f'final paths: {image_final_path} : { text_final_path}')

    #copy images in input-data to final-data/datetime
    mu.copy_folder_tree(image_initial_path, image_final_path)

    metadata_path = os.path.join(metadata_path, user_source_selected)
    text_folder_name = os.path.join(text_folder_name, user_source_selected)

    b_load_metadata = st.button("load image metadata")
    if b_load_metadata:

        df_metadata = load_metadata(metadata_path=metadata_path, metadata_file=metadata_file, image_final_path=image_final_path, image_final_folder=arc_folder_name)

        createVectorDB(df_metadata, vectordb_path, image_collection_name, text_folder_name, text_collection_name)

        archive_metadata(metadata_path, arc_folder_name, metadata_file)

        #mu.remove_files_folders(image_initial_path)
        #mu.remove_files_folders(text_folder_name)

if __name__=='__main__':
    execute()

