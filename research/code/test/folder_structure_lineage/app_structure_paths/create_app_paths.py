"""
app-root:
    /home/madhekar/temp/home-media-app
app-paths:
    /app/main
    /app-data/metadata
    /app-data/static-metadata/faces
    /app-data/static-metadata/locations/default
    /app-data/static-metadata/locations/user-specific
    /app-data/vectordb
    /data/final-data/img
    /data/final-data/txt
    /data/final-data/video
    /data/final-data/audio
    /data/input-data/img
    /data/input-data/txt
    /data/input-data/video
    /data/input-data/audio
    /data/input-data/error/img/duplicate
    /data/input-data/error/img/missing-data
    /data/input-data/error/img/quality
    /data/raw-data
    /data/train-data/img
    /data/train-data/txt
    /data/train-data/video
    /data/train-data/audio
    /models/faces_label_enc
    /models/faces_svc
    /models/faces_embeddings

"""
import os
import pathlib
import yaml
import shutil

def remove_folder_tree(folder):
    # Check if the folder exists before attempting to remove it
    if os.path.exists(folder):
        try:
            shutil.rmtree(folder)
            print(f"Folder '{folder}' and its contents removed successfully.")
        except OSError as e:
            print(f"Error: {e.filename} - {e.str.error}.")
    else:
        print(f"Folder '{folder}' does not exist.")

def create_folder_tree():

    with open('app_paths.yaml') as ifile:

        loaded_data = yaml.safe_load(ifile)
   
        root_folder = loaded_data['app-root']

        remove_folder_tree(root_folder)

        app_path_list = loaded_data['app-paths']

        for p in app_path_list:
            ap = os.path.join(root_folder, *p.split('/'))
            print(ap)
            pathlib.Path(ap).mkdir(parents= True, exist_ok= True)

        print(f"Folder '{root_folder}' and its contents created successfully.")

if __name__=='__main__':
    create_folder_tree()        