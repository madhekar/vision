import json
import pandas as pd
import yaml
import os

def load_metadata(metadata_path, metadata_file):
  data = []
  with open(os.path.join(metadata_path, metadata_file), mode="r") as f:
    for line in f:
        data.append(json.loads(line))

    df =  pd.DataFrame(data)
  return df

def populate_vectordb(df):   
  
  df_id = df["id"]
  df_metadata = df[["timestamp","lat","lon","loc","nam","txt"]].T.to_dict().values()
  df_url = df["url"]



  print(f'id: \n {df_id.head()} \n metadata: \n {df_metadata} \n url: \n {df_url.head()} ')

if __name__=='__main__':
    with open("metadata.yaml") as prop:
        dict = yaml.safe_load(prop)

        print("* * * * * * * * * * * Metadata Generator Properties * * * * * * * * * * * *")
        print(dict)
        print("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *")

    metadata_path = dict["metadata"]["metadata_path"]
    metadata_file = dict["metadata"]["metadata_file"]
    image_dir_path = dict["vectordb"]["vectordb_path"]   

    df_vector_database = load_metadata(metadata_path=metadata_path, metadata_file=metadata_file)

    populate_vectordb(df_vector_database)