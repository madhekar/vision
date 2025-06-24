
import pprint
import yaml
import streamlit as st



"""
metadata:
  static_metadata_path: /home/madhekar/work/home-media-app/data/app-data/static-metadata/
  static_metadata_file: static_locations.parquet
  missing_metadata_path: /home/madhekar/work/home-media-app/data/input-data/error/img/missing-data/
  missing_metadata_file: missing-metadata-wip.csv

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
"""
datapaths:
  raw_data_path: /home/madhekar/work/home-media-app/data/raw-data/
static-locations:
  location_metadata_path: /home/madhekar/work/home-media-app/data/static-data/static-locations/default
  user_location_metadata_path: /home/madhekar/work/home-media-app/data/static-data/static-locations/user-specific
  user_location_metadata_file: user-specific.csv
  user_draft_location_metadata_path_ext: draft
  user_draft_locations_metadata_file: user-specific-draft.csv

  missing_metadata_path: /home/madhekar/work/home-media-app/data/input-data-1/error/img/missing-data
  missing_metadata_file: missing-metadata-wip.csv
  missing_metdata_filter_file: missing-metdata-filter-wip.csv

  static_metadata_path: /home/madhekar/work/home-media-app/data/app-data/static-metadata
  static_metadata_file: static_locations.parquet

"""
@st.cache_resource
def static_metadata_config_load():
    with open("utils/config_util/static_metadata_conf.yaml") as prop:
        dict = yaml.safe_load(prop)

        pprint.pprint("* * *  dataload archiver properties * * *")
        pprint.pprint(dict)
        pprint.pprint("* * * * * * * * * * * * * * * * * * * * * *")

        raw_data_path = dict["datapaths"]["raw_data_path"]
        location_metadata_path = dict["static-locations"]["location_metadata_path"]
        user_location_metadata_path = dict["static-locations"]["user_location_metadata_path"]
        user_location_metadata_file = dict["static-locations"]["user_location_metadata_file"]

        user_draft_location_metadata_path_ext = dict["static-locations"]["user_draft_location_metadata_path_ext"]
        user_draft_location_metadata_file = dict["static-locations"]["user_draft_location_metadata_file"]

        missing_metadata_path = dict["static-locations"]["missing_metadata_path"]
        missing_metadata_file = dict["static-locations"]["missing_metadata_file"]
        missing_metadata_filter_file = dict["static-locations"]["missing_metadata_filter_file"]
        
        static_metadata_path = dict["static-locations"]["static_metadata_path"]
        static_metadata_file = dict["static-locations"]["static_metadata_file"]

    return (raw_data_path, 
            location_metadata_path, 
            user_location_metadata_path,  
            user_location_metadata_file,  
            user_draft_location_metadata_path_ext,
            user_draft_location_metadata_file,
            missing_metadata_path,
            missing_metadata_file,
            missing_metadata_filter_file,
            static_metadata_path, 
            static_metadata_file)    

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

        static_metadata_path = dict["static-metadata"]['static_metadata_path']
        static_metadata_file = dict['static-metadata'] ['static_metadata_file']
        
    return(
        image_dir_path,
        metadata_path,
        metadata_file,
        chunk_size,
        number_of_instances,
        openclip_finetuned, 
        static_metadata_path,
        static_metadata_file
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
        missing_metadata_filter_file = dict["missing-metadata"]["missing_metadata_filter_file"]
    return (
        input_image_path,
        missing_metadata_path,
        missing_metadata_file,
        missing_metadata_filter_file
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
    with open("utils/config_util/quality_conf.yaml") as prop:
        dict = yaml.safe_load(prop)

        pprint.pprint("* * * quality archiver properties * * *")
        pprint.pprint(dict)
        pprint.pprint("* * * * * * * * * * * * * * * * * * * * ")

        input_image_path = dict["quality"]["input_image_path"]
        archive_quality_path = dict["quality"]["archive_quality_path"]
        image_sharpness_threshold = dict["quality"]["image_sharpness_threshold"]
        image_quality_threshold = dict["quality"]["image_quality_threshold"]

    return (
        input_image_path,
        archive_quality_path,
        image_sharpness_threshold,
        image_quality_threshold
    )


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

        missing_metadata_path = dict["missing"]["base_path"]        
        missing_metadata_file = dict["missing"]["missing_metadata_file"]
        missing_metadata_filter_file = dict["missing"]["missing_metadata_filter_file"]

        metadata_file_path = dict["metadata"]["base_path"]

        static_metadata_file_path = dict["static-metadata"]["base_path"]
        vectordb_path = dict["vectordb"]["base_path"]

    return (
        raw_data_path,
        duplicate_data_path,
        quality_data_path,
        missing_metadata_path,
        missing_metadata_file,
        missing_metadata_filter_file,
        metadata_file_path,
        static_metadata_file_path,
        vectordb_path
    )


@st.cache_resource
def vectordb_config_load():
    with open("utils/config_util/preprocess_conf.yaml") as prop:
        dict = yaml.safe_load(prop)

        pprint.pprint("* * * Metadata Generator Properties * * *")
        pprint.pprint(dict)
        pprint.pprint("* * * * * * * * * * * * * * * * * * * * *")

        raw_data_path = dict["metadata"]["raw_data_path"]
        image_initial_path = dict["metadata"]["image_dir_path"]
        metadata_path = dict["metadata"]["metadata_path"]
        metadata_file = dict["metadata"]["metadata_file"]

        vectordb_path = dict['vectordb']["vectordb_path"]
        image_collection_name = dict['vectordb']["image_collection_name"]
        text_collection_name = dict['vectordb']["text_collection_name"]
        video_collection_name = dict['vectordb']["video_collection_name"]
        audio_collection_name = dict["vectordb"]["audio_collection_name"]

        text_dir_path = dict["vectordb"]["text_dir_path"]

        image_final_path = dict["prod"]["image_final_path"]
        text_final_path = dict["prod"]["text_final_path"]
        video_final_path = dict["prod"]["video_final_path"]
        audio_final_path = dict["prod"]["audio_final_path"]

    return(
        raw_data_path,
        image_initial_path,
        metadata_path,
        metadata_file,

        vectordb_path,
        image_collection_name,
        text_collection_name,
        video_collection_name,
        audio_collection_name,

        text_dir_path,

        image_final_path,
        text_final_path,
        video_final_path,
        audio_final_path
    )    

@st.cache_resource
def editor_config_load():
    with open("utils/config_util/editor_conf.yaml") as prop:
        dict = yaml.safe_load(prop)

        print("* * * Metadata Generator Properties * * *")
        pprint.pprint(dict)
        print("* * * * * * * * * * * * * * * * * * * * *")
        raw_data_path = dict["metadata"]["raw_data_path"]
        static_metadata_path = dict["metadata"]["static_metadata_path"]
        static_metadata_file = dict["metadata"]["static_metadata_file"]
        missing_metadata_path = dict["metadata"]["missing_metadata_path"]
        missing_metadata_file = dict["metadata"]["missing_metadata_file"]
        missing_metadata_filter_file = dict["metadata"]["missing_metadata_filter_file"]
        missing_metadata_edit_file = dict["metadata"]["missing_metadata_edit_file"]
        home_latitude = dict["metadata"]["home_latitude"]
        home_longitude = dict["metadata"]["home_longitude"]
    return (
        raw_data_path,
        static_metadata_path,
        static_metadata_file,
        missing_metadata_path,
        missing_metadata_file,
        missing_metadata_filter_file,
        missing_metadata_edit_file,
        home_latitude,
        home_longitude,
    )

@st.cache_resource
def search_config_load():
    with open("utils/config_util/search_conf.yaml") as prop:
        dict = yaml.safe_load(prop)

        pprint.pprint("* * * Metadata Generator Properties * * *")
        pprint.pprint(dict)
        pprint.pprint("* * * * * * * * * * * * * * * * * * * * *")

        vectordb_path = dict["vectordb"]["vectordb_path"]
        image_collection_name = dict["vectordb"]["image_collection_name"]
        text_collection_name = dict["vectordb"]["text_collection_name"]
        video_collection_name = dict["vectordb"]["video_collection_name"]
        audio_collection_name = dict["vectordb"]["audio_collection_name"]

    return (
        vectordb_path,
        image_collection_name,
        text_collection_name,
        video_collection_name,
        audio_collection_name,
    )

"""
static-metadata:
      faces_metadata_path: /home/madhekar/work/home-media-app/data/app-data/static-metadata/faces
      faces_of_people_parquet_path: /home/madhekar/work/home-media-app/data/app-data/static-metadata
      faces_of_people_parquet: image_people.parquet
image-data:
      input_image_path: /home/madhekar/work/home-media-app/data/input-data-1/img
model-path:
      faces_embbedings_path: /home/madhekar/work/home-media-app/models/faces_embbedings
      faces_embbedings: faces_embeddings_done_for_classes.npz
      faces_label_enc_path: /home/madhekar/work/home-media-app/models/faces_label_enc
      faces_label_enc: faces_label_enc.joblib
      faces_svc_path: /home/madhekar/work/home-media-app/models/faces_svc
      faces_svc: faces_model_svc.joblib

"""
@st.cache_resource
def faces_config_load():
    with open("utils/config_util/face_conf.yaml") as prop:
        dict = yaml.safe_load(prop)

        pprint.pprint("* * * Metadata Generator Properties * * *")
        pprint.pprint(dict)
        pprint.pprint("* * * * * * * * * * * * * * * * * * * * *")

        faces_metadata_path = dict["static-metadata"]["faces_metadata_path"]        
        faces_of_people_parquet_path = dict["static-metadata"]["faces_of_people_parquet_path"]
        faces_of_people_parquet = dict["static-metadata"]["faces_of_people_parquet"]

        input_image_path = dict["image-data"]["input_image_path"]
        
        faces_embbedings_path = dict["model-path"]["faces_embbedings_path"]
        faces_embbedings = dict["model-path"]["faces_embbedings"]
        faces_label_enc_path = dict["model-path"]["faces_label_enc_path"]
        faces_label_enc = dict["model-path"]["faces_label_enc"]
        faces_svc_path = dict["model-path"]["faces_svc_path"]
        faces_svc = dict["model-path"]["faces_svc"]


        return (
            faces_metadata_path,
            input_image_path,
            faces_embbedings_path,
            faces_embbedings,
            faces_label_enc_path,
            faces_label_enc,
            faces_svc_path,
            faces_svc,
            faces_of_people_parquet_path,
            faces_of_people_parquet
        )