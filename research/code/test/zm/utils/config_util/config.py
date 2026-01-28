import os
import pprint
import yaml
import streamlit as st

"""
metadata:
  static_metadata_path: /data/app-data/static-metadata/
  static_metadata_file: static_locations.parquet
  missing_metadata_path: /data/input-data/error/img/missing-data/
  missing_metadata_file: missing-metadata-wip.csv

datapaths:
  raw_data_path: '/data/raw-data'
  input_data_path: '/data/input-data'
  app_data_path: '/data/app-data'
  final_data_path: '/data/final-data'  

metadata:
  image_dir_path: /data/input-data/img/
  metadata_path: /data/app-data/metadata/
  metadata_file: metadata.json
  data_chunk_size: 10
  number_of_instances: 10

models:
  openclip_finetuned: /models/zeshaOpenClip/clip_finetuned.pth  

vectordb:
  vectordb_path: /data/app-data/vectordb/
  image_collection_name: multimodal_collection_images
  text_collection_name: multimodal_collection_texts
  video_collection_name: multimodal_collection_videos
  text_dir_path: /data/input-data/txt/

prod:
  image_final_path:  /data/final-data/img/
  text_final_path:  /data/final-data/txt/
  video_final_path:  /data/final-data/video/

dataload:
  input_image_path: '/data/input-data/img/'
  input_video_path: '/data/input-data/video/'
  input_txt_path: '/data/input-data/txt/'

addtrim:
  raw_data_path: '/data/raw-data/'
"""
"""
zmedia-setup:
  init_zmedia_path: /home/madhekar/work/home-media-app/app/zmedia-sample
  init_zmedia_file: zesha_media.zip

data-paths:
  raw_data_path: /data/raw-data
  input_data_path: /data/input-data
  app_data_path: /data/app-data
  final_data_path: /data/final-data
error-paths:
  img_dup_error_path: /data/input-data/error/img/duplicate
  img_qua_error_path: /data/input-data/error/img/quality
  img_mis_error_path: /data/input-data/error/img/missing-data
  video_dup_error_path: /data/input-data/error/video/duplicate
  video_qua_error_path: /data/input-data/error/video/quality
  video_mis_error_path: /data/input-data/error/video/missing-data
  text_dup_error_path: /data/input-data/error/txt/duplicate
  text_qua_error_path: /data/input-data/error/txt/quality
  text_mis_error_path: /data/input-data/error/txt/missing-data
  audio_dup_error_path: /data/input-data/error/audio/duplicate
  audio_qua_error_path: /data/input-data/error/audio/quality
  audio_mis_error_path: /data/input-data/error/audio/missing-data
input-paths:
  image_data_path: /data/input-data/img
  video_data_path: /data/input-data/video
  audio_data_path: /data/input-data/audio
  text_data_path: /data/input-data/txt
final-paths:
  image_data_path: /data/final-data/img
  video_data_path: /data/final-data/video
  audio_data_path: /data/final-data/audio
  text_data_path: /data/final-data/txt
"""
@st.cache_resource
def overview_config_load():
    dr,*_ = app_config_load()
    with open("utils/config_util/overview_conf.yaml") as prop:
        dict = yaml.safe_load(prop)

        pprint.pprint("* * *  overview archive properties * * *")
        pprint.pprint(dict)
        pprint.pprint("* * * * * * * * * * * * * * * * * * * * * *")

        raw_data_path = dict["data-paths"]["raw_data_path"]
        input_data_path = dict["data-paths"]["input_data_path"]
        app_data_path = dict["data-paths"]["app_data_path"]
        final_data_path = dict["data-paths"]["final_data_path"]

        image_data_path = dict["input-paths"]["image_data_path"]
        video_data_path = dict["input-paths"]["video_data_path"]
        text_data_path = dict["input-paths"]["text_data_path"]
        audio_data_path = dict["input-paths"]["audio_data_path"]

        final_image_data_path = dict["final-paths"]["image_data_path"]
        final_video_data_path = dict["final-paths"]["video_data_path"]
        final_text_data_path = dict["final-paths"]["text_data_path"]
        final_audio_data_path = dict["final-paths"]["audio_data_path"]
                        
        img_dup_error_path  = dict["error-paths"]["img_dup_error_path"]
        img_qua_error_path = dict["error-paths"]["img_qua_error_path"]
        img_mis_error_path = dict["error-paths"]["img_mis_error_path"]

        video_dup_error_path  = dict["error-paths"]["video_dup_error_path"]
        video_qua_error_path = dict["error-paths"]["video_qua_error_path"]
        video_mis_error_path = dict["error-paths"]["video_mis_error_path"]

        text_dup_error_path  = dict["error-paths"]["txt_dup_error_path"]
        text_qua_error_path = dict["error-paths"]["txt_qua_error_path"]
        text_mis_error_path = dict["error-paths"]["txt_mis_error_path"]

        audio_dup_error_path  = dict["error-paths"]["audio_dup_error_path"]
        audio_qua_error_path = dict["error-paths"]["audio_qua_error_path"]
        audio_mis_error_path = dict["error-paths"]["audio_mis_error_path"]

    return (
        os.path.join(dr, *raw_data_path.split(os.sep)[1:]),
        os.path.join(dr, *input_data_path.split(os.sep)[1:]),
        os.path.join(dr, *app_data_path.split(os.sep)[1:]),
        os.path.join(dr, *final_data_path.split(os.sep)[1:]),

        os.path.join(dr, *image_data_path.split(os.sep)[1:]),
        os.path.join(dr, *video_data_path.split(os.sep)[1:]),
        os.path.join(dr, *text_data_path.split(os.sep)[1:]),
        os.path.join(dr, *audio_data_path.split(os.sep)[1:]),

        os.path.join(dr, *final_image_data_path.split(os.sep)[1:]),
        os.path.join(dr, *final_video_data_path.split(os.sep)[1:]),
        os.path.join(dr, *final_text_data_path.split(os.sep)[1:]),
        os.path.join(dr, *final_audio_data_path.split(os.sep)[1:]),

        os.path.join(dr, *img_dup_error_path.split(os.sep)[1:]),
        os.path.join(dr, *img_qua_error_path.split(os.sep)[1:]),
        os.path.join(dr, *img_mis_error_path.split(os.sep)[1:]),

        os.path.join(dr, *video_dup_error_path.split(os.sep)[1:]),
        os.path.join(dr, *video_qua_error_path.split(os.sep)[1:]),
        os.path.join(dr, *video_mis_error_path.split(os.sep)[1:]),

        os.path.join(dr, *text_dup_error_path.split(os.sep)[1:]),
        os.path.join(dr, *text_qua_error_path.split(os.sep)[1:]),
        os.path.join(dr, *text_mis_error_path.split(os.sep)[1:]),

        os.path.join(dr, *audio_dup_error_path.split(os.sep)[1:]),
        os.path.join(dr, *audio_qua_error_path.split(os.sep)[1:]),
        os.path.join(dr, *audio_mis_error_path.split(os.sep)[1:]),
    )

"""

"""
@st.cache_resource
def dataload_config_load():
    dr,*_ = app_config_load()
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

    return (
        os.path.join(dr, *raw_data_path.split(os.sep)[1:]),
        os.path.join(dr, *input_image_path.split(os.sep)[1:]),
        os.path.join(dr, *input_txt_path.split(os.sep)[1:]),
        os.path.join(dr, *input_video_path.split(os.sep)[1:]),
        os.path.join(dr, *input_audio_path.split(os.sep)[1:])
    )

"""

"""
@st.cache_resource
def static_metadata_config_load():
    dr,*_ = app_config_load()
    with open("utils/config_util/static_metadata_conf.yaml") as prop:
        dict = yaml.safe_load(prop)

        pprint.pprint("* * *  dataload archive properties * * *")
        pprint.pprint(dict)
        pprint.pprint("* * * * * * * * * * * * * * * * * * * * * *")

        raw_data_path = dict["datapaths"]["raw_data_path"]

        static_metadata_path = dict['static-metadata']['static_metadata_path']

        faces_metadata_path = dict['static-faces']['faces_metadata_path']
  
        filter_metadata_path = dict['static-filter']['filter_metadata_path']

        default_location_metadata_path = dict["static-locations"]["default_location_metadata_path"]
        user_location_metadata_path = dict["static-locations"]["user_location_metadata_path"]
        user_location_metadata_file = dict["static-locations"]["user_location_metadata_file"]
        final_user_location_metadata_file = dict["static-locations"]["final_user_location_metadata_file"]

        missing_metadata_path = dict["missing-metadata"]["missing_metadata_path"]
        missing_metadata_file = dict["missing-metadata"]["missing_metadata_file"]
        missing_metadata_filter_file = dict["missing-metadata"]["missing_metadata_filter_file"]


    return (
        os.path.join(dr, *raw_data_path.split(os.sep)[1:]),
        os.path.join(dr, *static_metadata_path.split(os.sep)[1:]),
        os.path.join(dr, *faces_metadata_path.split(os.sep)[1:]),
        os.path.join(dr, *filter_metadata_path.split(os.sep)[1:]),
        os.path.join(dr, *default_location_metadata_path.split(os.sep)[1:]),
        os.path.join(dr, *user_location_metadata_path.split(os.sep)[1:]),
        user_location_metadata_file,
        final_user_location_metadata_file,
        os.path.join(dr, *missing_metadata_path.split(os.sep)[1:]),
        missing_metadata_file,
        missing_metadata_filter_file,
    )    

"""

"""
@st.cache_resource
def preprocess_config_load():
    dr,*_ = app_config_load()
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
        os.path.join(dr, *image_dir_path.split(os.sep)[1:]),
        os.path.join(dr, *metadata_path.split(os.sep)[1:]),
        metadata_file,
        chunk_size,
        number_of_instances,
        openclip_finetuned, 
        os.path.join(dr, *static_metadata_path.split(os.sep)[1:]),
        static_metadata_file
    )    

"""

"""
@st.cache_resource
def missing_metadata_config_load():
    dr,*_ = app_config_load()
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
        os.path.join(dr, *input_image_path.split(os.sep)[1:]),
        os.path.join(dr, *missing_metadata_path.split(os.sep)[1:]),
        missing_metadata_file,
        missing_metadata_filter_file
    )

"""

"""
@st.cache_resource
def dedup_config_load():
    dr,*_ = app_config_load()
    with open("utils/config_util/dedup_conf.yaml") as prop:
        dict = yaml.safe_load(prop)

        pprint.pprint("* * *  duplicate archive properties * * *")
        pprint.pprint(dict)
        pprint.pprint("* * * * * * * * * * * * * * * * * * * * * *")

        input_image_path = dict["duplicate"]["input_image_path"]
        archive_dup_path = dict["duplicate"]["archive_dup_path"]
    return (
        os.path.join(dr, *input_image_path.split(os.sep)[1:]),
        os.path.join(dr, *archive_dup_path.split(os.sep)[1:])
    )

"""

"""
@st.cache_resource
def image_quality_config_load():
    dr,*_ = app_config_load()
    with open("utils/config_util/quality_conf.yaml") as prop:
        dict = yaml.safe_load(prop)

        pprint.pprint("* * * quality archive properties * * *")
        pprint.pprint(dict)
        pprint.pprint("* * * * * * * * * * * * * * * * * * * * ")

        input_image_path = dict["quality"]["input_image_path"]
        archive_quality_path = dict["quality"]["archive_quality_path"]
        image_quality_threshold = dict["quality"]["image_quality_threshold"]
    return (
        os.path.join(dr, *input_image_path.split(os.sep)[1:]),
        os.path.join(dr, *archive_quality_path.split(os.sep)[1:]),
        image_quality_threshold,
    )

"""

"""
@st.cache_resource
def data_validation_config_load():
    dr,*_ = app_config_load()
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
        os.path.join(dr, *raw_data_path.split(os.sep)[1:]),
        os.path.join(dr, *duplicate_data_path.split(os.sep)[1:]),
        os.path.join(dr, *quality_data_path.split(os.sep)[1:]),
        os.path.join(dr, *missing_metadata_path.split(os.sep)[1:]),
        missing_metadata_file,
        missing_metadata_filter_file,
        os.path.join(dr, *metadata_file_path.split(os.sep)[1:]),
        os.path.join(dr, *static_metadata_file_path.split(os.sep)[1:]),
        os.path.join(dr, *vectordb_path.split(os.sep)[1:])
    )

"""

"""
@st.cache_resource
def vectordb_config_load():
    dr, *_ = app_config_load()
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
        os.path.join(dr, *raw_data_path.split(os.sep)[1:]),
        os.path.join(dr, *image_initial_path.split(os.sep)[1:]),
        os.path.join(dr, *metadata_path.split(os.sep)[1:]),
        metadata_file,

        os.path.join(dr, *vectordb_path.split(os.sep)[1:]),
        image_collection_name,
        text_collection_name,
        video_collection_name,
        audio_collection_name,

        os.path.join(dr, *text_dir_path.split(os.sep)[1:]),

        os.path.join(dr, *image_final_path.split(os.sep)[1:]),
        os.path.join(dr, *text_final_path.split(os.sep)[1:]),
        os.path.join(dr, *video_final_path.split(os.sep)[1:]),
        os.path.join(dr, *audio_final_path.split(os.sep)[1:])
    )    
"""

"""
@st.cache_resource
def editor_config_load():
    dr, *_ = app_config_load()
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
        batch_size_max = dict["metadata"]["batch_size_max"]
        row_size_preset = dict["metadata"]["row_size_preset"]
    return (
        os.path.join(dr, *raw_data_path.split(os.sep)[1:]),
        os.path.join(dr, *static_metadata_path.split(os.sep)[1:]),
        static_metadata_file,
        os.path.join(dr, *missing_metadata_path.split(os.sep)[1:]),
        missing_metadata_file,
        missing_metadata_filter_file,
        missing_metadata_edit_file,
        home_latitude,
        home_longitude,
        batch_size_max,
        row_size_preset
    )
"""

"""
@st.cache_resource
def search_config_load():
    dr, *_ = app_config_load()
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
        os.path.join(dr, *vectordb_path.split(os.sep)[1:]),
        image_collection_name,
        text_collection_name,
        video_collection_name,
        audio_collection_name,
    )

"""

static-metadata:
      faces_metadata_path: /data/app-data/static-metadata/faces/training/images
model-path:
      faces_label_enc_path: /models/faces_label_enc
      faces_label_enc: faces_label_enc.joblib
      faces_svc_path: /models/faces_svc
      faces_svc: faces_model_svc.joblib

"""
@st.cache_resource
def faces_config_load():

    # global app attributes
    dr,*_ = app_config_load()

    with open("utils/config_util/face_conf.yaml") as prop:
        dict = yaml.safe_load(prop)

        pprint.pprint("* * * Metadata Generator Properties * * *")
        pprint.pprint(dict)
        pprint.pprint("* * * * * * * * * * * * * * * * * * * * *")

        faces_metadata_path = dict["static-metadata"]["faces_metadata_path"]        
        
        faces_label_enc_path = dict["model-path"]["faces_label_enc_path"]
        faces_label_enc = dict["model-path"]["faces_label_enc"]
        faces_svc_path = dict["model-path"]["faces_svc_path"]
        faces_svc = dict["model-path"]["faces_svc"]

        return (
            os.path.join(dr, *faces_metadata_path.split(os.sep)[1:]),
            os.path.join(dr, *faces_label_enc_path.split(os.sep)[1:]),
            faces_label_enc,
            os.path.join(dr, *faces_svc_path.split(os.sep)[1:]),
            faces_svc,
        )
    
"""    

"""
@st.cache_resource
def filer_config_load():
    dr, *_ = app_config_load()
    with open("utils/config_util/filter_conf.yaml") as prop:
        dict = yaml.safe_load(prop)

        pprint.pprint("* * *  filter model archive properties * * *")
        pprint.pprint(dict)
        pprint.pprint("* * * * * * * * * * * * * * * * * * * * * *")

        base_model_path = dict["img-filter-model"]["filter_model_path"]
        data_path = dict["img-filter-model"]["data_path"]
        filter_model_name = dict["img-filter-model"]["filter_model_name"]
        filter_model_classes = dict["img-filter-model"]["filter_model_classes"]
        image_size = dict["img-filter-model"]["image_size"]
        batch_size = dict["img-filter-model"]["batch_size"]
        num_epocs = dict["img-filter-model"]["number_of_epocs"]

    return (
        os.path.join(dr, *base_model_path.split(os.sep)[1:]),
        os.path.join(dr, *data_path.split(os.sep)[1:]),
        filter_model_name,
        filter_model_classes,
        image_size,
        batch_size,
        num_epocs
    )

'''
app-root:
   app_root_path: /home/madhekar/work
final-archive:
   final_image_path: home-media-app/data/final-data/img
   final_video_path: home-media-app/data/final-data/video
   final_text_path: home-media-app/data/final-data/txt
   final_audio_path: home-media-app/data/final-data/audio
model-archive:
   model_path: home-media-app/models
appdata-archive:
   appdata_path: home-media-app/data/app-data
'''
@st.cache_resource
def archive_config_load():

    with open("utils/config_util/archive_conf.yaml") as prop:
        dict = yaml.safe_load(prop)

        pprint.pprint("* * * Archive Properties * * *")
        pprint.pprint(dict)
        pprint.pprint("* * * * * * * * * * * * * * * * * * * * *")

        data_root = dict["app-root"]["app_root_path"]
        final_image_path = dict["final-archive"]["final_image_path"]
        final_video_path = dict["final-archive"]["final_video_path"]
        final_text_path = dict["final-archive"]["final_text_path"]
        final_audio_path = dict["final-archive"]["final_audio_path"]
        
        model_path = dict["model-archive"]["model_path"]

        appdata_path = dict["appdata-archive"]["appdata_path"]

        return(
            data_root,
            final_image_path,
            final_video_path,
            final_text_path,
            final_audio_path,
            model_path,
            appdata_path
        )
"""    
      
"""
@st.cache_resource
def app_config_load():
    with open("utils/config_util/app_conf.yaml") as prop:
        dict = yaml.safe_load(prop)

        pprint.pprint("* * * App Properties * * *")
        pprint.pprint(dict)
        pprint.pprint("* * * * * * * * * * * * * * * * * * * * *")

        data_root = dict["app-config"]["appdata_root_path"]
        app_root = dict["app-config"]["approot_path"]

        return(
            data_root,
            app_root
        )

"""

"""
@st.cache_resource
def setup_config_load():
    dr, *_ = app_config_load()
    ap, dp, mp = [], [], []
    with open("utils/config_util/setup_conf.yaml") as prop:
        dict = yaml.safe_load(prop)

        pprint.pprint("* * * Metadata Generator Properties * * *")
        pprint.pprint(dict)
        pprint.pprint("* * * * * * * * * * * * * * * * * * * * *")

        for app_pth in dict["app_paths"]:
            pth = os.path.join(dr, app_pth) # *app_pth.split(os.sep)[1:])
            ap.append(pth)

        for data_pth in dict["data_paths"]:
            pth = os.path.join(dr, data_pth) # *data_pth.split(os.sep)[1:])
            dp.append(pth)

        for model_pth in dict["model_paths"]:
            pth = os.path.join(dr, model_pth) #*model_pth.split(os.sep)[1:])
            mp.append(pth)    

    return (ap, dp, mp)            