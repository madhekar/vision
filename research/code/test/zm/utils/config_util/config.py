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
"""
@st.cache_resource
def overview_config_load():
    dr,*_ = app_config_load()
    with open("utils/config_util/overview_conf.yaml") as prop:
        dict = yaml.safe_load(prop)

        pprint.pprint("* * *  overview archive properties * * *")
        pprint.pprint(dict)
        pprint.pprint("* * * * * * * * * * * * * * * * * * * * * *")

        raw_data_path = dict["datapaths"]["raw_data_path"]
        input_data_path = dict["datapaths"]["input_data_path"]
        app_data_path = dict["datapaths"]["app_data_path"]
        final_data_path = dict["datapaths"]["final_data_path"]

    return (
        os.path.join(dr, *raw_data_path.split(os.sep)[1:]),
        os.path.join(dr, *input_data_path.split(os.sep)[1:]),
        os.path.join(dr, *app_data_path.split(os.sep)[1:]),
        os.path.join(dr, *final_data_path.split(os.sep)[1:])
    )

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
datapaths:
  raw_data_path: /data/raw-data/
static-metadata:
  static_metadata_path: /data/app-data/static-metadata  
static-faces: 
  faces_metadata_path: /data/app-data/static-metadata/faces
static-filter:
  filter_metadata_path: /models/image_classify_filter/training  
static-locations:
  default_location_metadata_path: /data/app-data/static-metadata/locations/default
  user_location_metadata_path: /data/app-data/static-metadata/locations/user-specific
  user_location_metadata_file: user-specific.csv
  final_user_location_metadata_file: static_locations.parquet
missing-metadata:  
  missing_metadata_path: /data/input-data/error/img/missing-data
  missing_metadata_file: missing-metadata-wip.csv
  missing_metadata_filter_file: missing-metadata-filter-wip.csv

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
brisque-model:
  brisque_model_config_path: '/models/brisque'
  brisque_model_live_file: 'brisque_model_live.yml'
  brisque_range_live_file: 'brisque_range_live.yml'
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
      faces_metadata_path: /data/app-data/static-metadata/faces
      faces_of_people_parquet_path: /data/app-data/static-metadata
      faces_of_people_parquet: image_people.parquet
image-data:
      input_image_path: /data/input-data/img
model-path:
      faces_embeddings_path: /models/faces_embeddings
      faces_embeddings: faces_embeddings_done_for_classes.npz
      faces_label_enc_path: /models/faces_label_enc
      faces_label_enc: faces_label_enc.joblib
      faces_svc_path: /models/faces_svc
      faces_svc: faces_model_svc.joblib
 
"""
@st.cache_resource
def faces_config_load():
    dr,*_ = app_config_load()
    with open("utils/config_util/face_conf.yaml") as prop:
        dict = yaml.safe_load(prop)

        pprint.pprint("* * * Metadata Generator Properties * * *")
        pprint.pprint(dict)
        pprint.pprint("* * * * * * * * * * * * * * * * * * * * *")

        faces_metadata_path = dict["static-metadata"]["faces_metadata_path"]        
        faces_of_people_parquet_path = dict["static-metadata"]["faces_of_people_parquet_path"]
        faces_of_people_parquet = dict["static-metadata"]["faces_of_people_parquet"]

        input_image_path = dict["image-data"]["input_image_path"]
        
        faces_embeddings_path = dict["model-path"]["faces_embeddings_path"]
        faces_embeddings = dict["model-path"]["faces_embeddings"]
        faces_label_enc_path = dict["model-path"]["faces_label_enc_path"]
        faces_label_enc = dict["model-path"]["faces_label_enc"]
        faces_svc_path = dict["model-path"]["faces_svc_path"]
        faces_svc = dict["model-path"]["faces_svc"]


        return (
            os.path.join(dr, *faces_metadata_path.split(os.sep)[1:]),
            os.path.join(dr, *input_image_path.split(os.sep)[1:]),
            os.path.join(dr, *faces_embeddings_path.split(os.sep)[1:]),
            faces_embeddings,
            os.path.join(dr, *faces_label_enc_path.split(os.sep)[1:]),
            faces_label_enc,
            os.path.join(dr, *faces_svc_path.split(os.sep)[1:]),
            faces_svc,
            os.path.join(dr, *faces_of_people_parquet_path.split(os.sep)[1:]),
            faces_of_people_parquet
        )
    
"""    
filter-model:
  filter_model_path: /models/image_classify_filter
  train_data_path: /models/image_classify_filter/training
  validation_data_path: /models/image_classify_filter/validation
  testing_data_path: /models/image_classify_filter/testing
  testing_data_map_file: map.json  
  filter_model_name: filter_images_Model.keras
  filter_model_classes: class_names.joblib
  image_size: (224, 224)
  batch_size: 32
      
"""

@st.cache_resource
def filer_config_load():
    dr, *_ = app_config_load()
    with open("utils/config_util/filter_conf.yaml") as prop:
        dict = yaml.safe_load(prop)

        pprint.pprint("* * *  filter model archive properties * * *")
        pprint.pprint(dict)
        pprint.pprint("* * * * * * * * * * * * * * * * * * * * * *")

        base_model_path = dict["filter-model"]["filter_model_path"]
        train_data_path = dict["filter-model"]["train_data_path"]
        validation_data_path = dict["filter-model"]["validation_data_path"]
        testing_data_path = dict["filter-model"]["testing_data_path"]
        testing_data_map_file = dict["filter-model"]["testing_data_map_file"]
        filter_model_name = dict["filter-model"]["filter_model_name"]
        filter_model_classes = dict["filter-model"]["filter_model_classes"]
        image_size = dict["filter-model"]["image_size"]
        batch_size = dict["filter-model"]["batch_size"]

    return (
        os.path.join(dr, *base_model_path.split(os.sep)[1:]),
        os.path.join(dr, *train_data_path.split(os.sep)[1:]),
        os.path.join(dr, *validation_data_path.split(os.sep)[1:]),
        os.path.join(dr, *testing_data_path.split(os.sep)[1:]),
        testing_data_map_file,
        filter_model_name,
        filter_model_classes,
        image_size,
        batch_size,
    )

"""    
app-config:
      appdata_root_path: 
      approot_path: /home/madhekar/work/vision/research/code/test/multipage_app
zmedia-setup:
  init_zmedia_path: ../zmedia-sample
  init_zmedia_file: zesha_media.zip  
  raw_data_path: data/raw_data
      
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