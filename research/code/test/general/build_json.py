import json
import os

img_root = "/home/madhekar/work/home-media-app/data/app-data/qa/face-detection/images"
qa_metadata_path = "/home/madhekar/work/home-media-app/data/app-data/qa/face-detection"
qa_metadata_file = "face-detection-qa.json"

def build_json(img_root):
   files = os.listdir(img_root)
   test_data = [{"img": os.path.join(img_root, img), "text": ""} for img in files]
   with open(os.path.join(qa_metadata_path, qa_metadata_file), "w") as f:
      json.dump(test_data, f, indent=2)

build_json(img_root)