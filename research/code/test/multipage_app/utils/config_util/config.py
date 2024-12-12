
import pprint
import yaml
import streamlit as st

@st.cache_resource
def editor_config_load():
    with open("utils/config_util/editor_conf.yaml") as prop:
        dict = yaml.safe_load(prop)

        print("* * * Metadata Generator Properties * * *")
        pprint.pprint(dict)
        print("* * * * * * * * * * * * * * * * * * * * *")
        static_metadata_path = dict["metadata"]["static_metadata_path"]
        static_metadata_file = dict["metadata"]["static_metadata_file"]
        missing_metadata_path = dict["metadata"]["missing_metadata_path"]
        missing_metadata_file = dict["metadata"]["missing_metadata_file"]
        sqlite_database_path = dict["metadata"]["sqlite_database_path"]
        sqlite_database_name = dict["metadata"]["sqlite_database_name"]
    return (
        static_metadata_path,
        static_metadata_file,
        missing_metadata_path,
        missing_metadata_file,
        sqlite_database_path,
        sqlite_database_name,
    )

"""

datapaths:
  raw_data_path: '/home/madhekar/work/home-media-app/data/raw-data'
  input_data_path: '/home/madhekar/work/home-media-app/data/input-data'
  app_data_path: '/home/madhekar/work/home-media-app/data/app-data'
  final_data_path: '/home/madhekar/work/home-media-app/data/final-data'  

metadata:
  image_dir_path: /home/madhekar/work/home-media-app/data/input-data/img/
  metadata_path: /home/madhekar/work/home-media-app/data/app-data/metadata/
  metadata_file: metadata.json
  data_chunk_size: 10
  number_of_instances: 10

models:
  openclip_finetuned: /home/madhekar/work/home-media-app/models/zeshaOpenClip/clip_finetuned.pth  

vectordb:
  vectordb_path: /home/madhekar/work/home-media-app/data/app-data/vectordb/
  image_collection_name: multimodal_collection_images
  text_collection_name: multimodal_collection_texts
  video_collection_name: multimodal_collection_videos
  text_dir_path: /home/madhekar/work/home-media-app/data/input-data/txt/

prod:
  image_final_path:  /home/madhekar/work/home-media-app/data/final-data/img/
  text_final_path:  /home/madhekar/work/home-media-app/data/final-data/txt/
  video_final_path:  /home/madhekar/work/home-media-app/data/final-data/video/

dataload:
  input_image_path: '/home/madhekar/work/home-media-app/data/input-data/img/'
  input_video_path: '/home/madhekar/work/home-media-app/data/input-data/video/'
  input_txt_path: '/home/madhekar/work/home-media-app/data/input-data/txt/'

addtrim:
  raw_data_path: '/home/madhekar/work/home-media-app/data/raw-data/'
"""

@st.cache_resource
def overview_config_load():
    with open("utils/config_util/overview_conf.yaml") as prop:
        dict = yaml.safe_load(prop)

        pprint.pprint("* * *  overview archiver properties * * *")
        pprint.pprint(dict)
        pprint.pprint("* * * * * * * * * * * * * * * * * * * * * *")

        raw_data_path = dict["datapaths"]["raw_data_path"]
        input_data_path = dict["datapaths"]["input_data_path"]
        app_data_path = dict["datapaths"]["app_data_path"]
        final_data_path = dict["datapaths"]["final_data_path"]


    return (raw_data_path, input_data_path, app_data_path, final_data_path)

@st.cache_resource
def dataload_config_load():
    with open("utils/config_util/dataload_conf.yaml") as prop:
        dict = yaml.safe_load(prop)

        pprint.pprint("* * *  dataload archiver properties * * *")
        pprint.pprint(dict)
        pprint.pprint("* * * * * * * * * * * * * * * * * * * * * *")

        raw_data_path = dict["dataload"]["raw_data_path"]
        input_image_path = dict["dataload"]["input_image_path"]
        input_txt_path = dict["dataload"]["input_txt_path"]
        input_video_path = dict["dataload"]["input_video_path"]
        input_audio_path = dict["dataload"]["input_audio_path"]

    return (raw_data_path, input_image_path, input_txt_path, input_video_path, input_audio_path)



@st.cache_resource
def preprocess_config_load():
    with open("utils/config_util/preprocess_conf.yaml") as prop:
        dict = yaml.safe_load(prop)

        pprint.pprint("* * * Metadata Generator Properties * * *")
        pprint.pprint(dict)
        pprint.pprint("* * * * * * * * * * * * * * * * * * * * *")

        image_dir_path = dict["metadata"]["image_dir_path"]
        metadata_path = dict["metadata"]["metadata_path"]
        metadata_file = dict["metadata"]["metadata_file"]
        chunk_size = dict["metadata"]["data_chunk_size"]
        number_of_instances = dict["metadata"]["number_of_instances"]

        openclip_finetuned = dict["models"]["openclip_finetuned"] 
        
    return(
        image_dir_path,
        metadata_path,
        metadata_file,
        chunk_size,
        number_of_instances,
        openclip_finetuned
    )    

@st.cache_resource
def missing_metadata_config_load():
    with open("utils/config_util/missing_metadata_conf.yaml") as prop:
        dict = yaml.safe_load(prop)

        pprint.pprint("* * * Missing Metadata Properties * * *")
        pprint.pprint(dict)
        pprint.pprint("* * * * * * * * * * * * * * * * * * * *")
        input_image_path = dict["missing-metadata"]["input_image_path"]
        missing_metadata_path = dict["missing-metadata"]["missing_metadata_path"]
        missing_metadata_file = dict["missing-metadata"]["missing_metadata_file"]
    return (
        input_image_path,
        missing_metadata_path,
        missing_metadata_file,
    )

@st.cache_resource
def dedup_config_load():
    with open("utils/config_util/dedup_conf.yaml") as prop:
        dict = yaml.safe_load(prop)

        pprint.pprint("* * *  duplicate archiver properties * * *")
        pprint.pprint(dict)
        pprint.pprint("* * * * * * * * * * * * * * * * * * * * * *")

        input_image_path = dict["duplicate"]["input_image_path"]
        archive_dup_path = dict["duplicate"]["archive_dup_path"]
    return (
        input_image_path,
        archive_dup_path
    )

@st.cache_resource
def image_quality_config_load():
    with open("utils/config_util/image_quality_conf.yaml") as prop:
        dict = yaml.safe_load(prop)

        pprint.pprint("* * * quality archiver properties * * *")
        pprint.pprint(dict)
        pprint.pprint("* * * * * * * * * * * * * * * * * * * * ")

        input_image_path = dict["quality"]["input_image_path"]
        archive_quality_path = dict["quality"]["archive_quality_path"]
        image_sharpness_threshold = dict["quality"]["image_sharpness_threshold"]
    return (input_image_path, archive_quality_path, image_sharpness_threshold)


@st.cache_resource
def data_validation_config_load():
    with open("utils/config_util/data_validation_conf.yaml") as prop:
        dict = yaml.safe_load(prop)

        pprint.pprint("* * * Data Validation Properties * * *")
        pprint.pprint(dict)
        pprint.pprint("* * * * * * * * * * * * * * * * * * * * *")

        raw_data_path = dict["raw-data"]["base_path"]
        duplicate_data_path = dict["duplicate"]["base_path"]
        quality_data_path = dict["quality"]["base_path"]
        missing_data_path = dict["missing"]["base_path"]
        metadata_file_path = dict["metadata"]["base_path"]
        static_metadata_file_path = dict["static-metadata"]["base_path"]
        vectordb_path = dict["vectordb"]["base_path"]

    return (
        raw_data_path,
        duplicate_data_path,
        quality_data_path,
        missing_data_path,
        metadata_file_path,
        static_metadata_file_path,
        vectordb_path,
    )

@st.cache_resource
def vectordb_config_load():
    with open("utils/config_util/preprocess_conf.yaml") as prop:
        dict = yaml.safe_load(prop)

        pprint.pprint("* * * Metadata Generator Properties * * *")
        pprint.pprint(dict)
        pprint.pprint("* * * * * * * * * * * * * * * * * * * * *")

        image_initial_path = dict["metadata"]["image_dir_path"]
        metadata_path = dict["metadata"]["metadata_path"]
        metadata_file = dict["metadata"]["metadata_file"]

        vectordb_path = dict['vectordb']["vectordb_path"]
        image_collection_name = dict['vectordb']["image_collection_name"]
        text_collection_name = dict['vectordb']["text_collection_name"]
        video_collection_name = dict['vectordb']["video_collection_name"]

        text_dir_path = dict["vectordb"]["video_collection_name"]

        image_final_path = dict["prod"]["image_final_path"]
        text_final_path = dict["prod"]["text_final_path"]
        video_final_path = dict["prod"]["video_final_path"]

    return(
        image_initial_path,
        metadata_path,
        metadata_file,

        vectordb_path,
        image_collection_name,
        text_collection_name,
        video_collection_name,

        text_dir_path,

        image_final_path,
        text_final_path,
        video_final_path
    )    