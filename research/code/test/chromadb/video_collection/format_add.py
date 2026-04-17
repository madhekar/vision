import os
import json
import pandas as pd


# handle new creation on metadata file from scratch
def load_metadata(metadata_path, metadata_file, image_final_path, image_final_folder):
    data = []
    with open(os.path.join(metadata_path, metadata_file), mode="r") as f:
        for line in f:
            data.append(json.loads(line))

        df = pd.DataFrame(data)

        # df["uri"] = df["uri"].str.replace(
        #     "input-data/img",
        #     "final-data/img" #+ image_final_path,
        # )
        print(df)
    return df


vmpath ="/mnt/zmdata/home-media-app/data/app-data/metadata/Berkeley/"
vmfile ="video_metadata.json"

load_metadata(vmpath, vmfile, "", "")