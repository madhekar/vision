"""
/home-media-app/app/main
/home-media-app/app-data/metadata
/home-media-app/app-data/static-metadata/faces
/home-media-app/app-data/static-metadata/locations/default
/home-media-app/app-data/static-metadata/locations/user-specific
/home-media-app/app-data/vectordb
/home-media-app/data/final-data/img
/home-media-app/data/final-data/txt
/home-media-app/data/final-data/video
/home-media-app/data/final-data/audio
/home-media-app/data/input-data/img
/home-media-app/data/input-data/txt
/home-media-app/data/input-data/video
/home-media-app/data/input-data/audio
/home-media-app/data/input-data/error/img/duplicate
/home-media-app/data/input-data/error/img/missing-data
/home-media-app/data/input-data/error/img/quality
/home-media-app/data/raw-data
/home-media-app/data/train-data/img
/home-media-app/data/train-data/txt
/home-media-app/data/train-data/video
/home-media-app/data/train-data/audio
/home-media-app/models/faces_label_enc
/home-media-app/models/faces_svc
/home-media-app/models/faces_embeddings

"""
import os
import pathlib
import yaml

with open('app_paths.yaml') as ifile:
    loded_data = yaml.safe_load(ifile)

    root_folder = loded_data['app-root']
    app_path_list = loded_data['app-paths']

    for p in app_path_list:
        ap = os.path.join(root_folder, *p.split('/'))
        print(ap)
        pathlib.Path(ap).mkdir(parents= True, exist_ok= True)

