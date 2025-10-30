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
def folder_setup(root):
    
    # load config
    
    
    # create app folders

    #create data folders

    #create model folders