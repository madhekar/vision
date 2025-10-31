import os
import yaml
from pathlib import Path

'''
├── app
│   └── main
├── data
│   ├── app-data
│   │   ├── metadata
│   │   ├── qa
│   │   │   └── face-detection
│   │   │       └── images
│   │   ├── static-metadata
│   │   │   ├── faces
│   │   │   └── locations
│   │   │       ├── default
│   │   │       └── user-specific
│   │   └── vectordb
│   ├── final-data
│   │   ├── audio
│   │   ├── img
│   │   ├── txt
│   │   └── video
│   ├── input-data
│   │   ├── audio
│   │   ├── error
│   │   │   └── img
│   │   │       ├── duplicate
│   │   │       ├── missing-data
│   │   │       └── quality
│   │   ├── img
│   │   ├── txt
│   │   └── video
│   └── raw-data
└── models
    ├── faces_embeddings
    ├── faces_label_enc
    ├── faces_svc
    └── image_classify_filter

'''
def create_path_hirarchy(pth):
    fpth = Path(pth)
    try:
        fpth.mkdir(parents=True, exist_ok=True)
        print(f" created: {fpth}")
    except OSError as e:
        print(f"error creating: {fpth} with exception: {e}")    

def folder_setup(root, ap, dp, mp):
    
    # create app folders
    for p in ap:
      create_path_hirarchy(p)

    #create data folders

    for p in dp:
        create_path_hirarchy(p)

    #create model folders
    for p in mp:
        create_path_hirarchy(p)