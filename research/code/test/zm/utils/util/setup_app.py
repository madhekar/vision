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

    app_paths = [
                 "app/main"
                ]
    data_paths = [
                  "data/app-data/metadata",
                  "data/app-data/qa/face-detection/images",
                  "data/app-data/static-metadata/faces",
                  "data/app-data/static-metadata/locations/default",
                  "data/app-data/static-metadata/locations/user-specific",
                  "data/app-data/vectordb",
                  "data/input-data/audio",
                  "data/input-data/img",
                  "data/input-data/txt",
                  "data/input-data/video",
                  "data/input-data/error/img/duplicate",
                  "data/input-data/error/img/missing-data",
                  "data/input-data/error/img/quality",
                  "data/raw-data"
                   ]
    model_paths = [                   
                   "models/faces-embeddings",
                   "models/faces_label_enc",
                   "models/faces_svc",
                   "models/image_classify_filter"
                   ]
    # create app folders

    #create data folders

    #create model folders