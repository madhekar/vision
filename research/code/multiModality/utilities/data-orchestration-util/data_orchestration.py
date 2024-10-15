
import os
import yaml
import pprint


def build_orch_structure():
    """
        idx, user-folder-name, state,
    """

def extract_user_raw_data_folders(path):
   next(os.walk(path))[1]

def config_load():
    with open("data_orchestration_conf.yaml") as prop:
        dict = yaml.safe_load(prop)

        pprint.pprint("* * * * * * * * * * * Data Orchestration Properties * * * * * * * * * * * *")
        pprint.pprint(dict)
        pprint.pprint("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * ")
        raw_data_path = dict["raw-data"]["base_path"]
        duplicate_data_path = dict["duplicate"]["base_path"]
        quality_data_path = dict["quality"]["base_path"]
        missing_data_path = dict["missing"]["base_path"]
        metadata_file_path = dict["metadata"]["base_path"]
        static_metadata_file_path = dict["static-metadata"]["base_path"]
        vectordb_path = dict["vectordb"]["base_path"]
    return (
        input_image_path,
        missing_metadata_path,
        missing_metadata_file,
    )

if __name__ == '__main__':
    config_load()
  