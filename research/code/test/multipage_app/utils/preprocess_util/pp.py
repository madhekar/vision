import os
import uuid
import glob
import asyncio
from utils.preprocess_util import awaitUtil
from utils.preprocess_util import entities as en
from utils.preprocess_util import LLM
import aiofiles
import json
import pandas as pd
import streamlit as st
from utils.config_util import config
from utils.util import location_util as lu
from utils.util import model_util as mu

d_latitude, d_longitude = 32.96887205555556, -117.18414305555557


# init LLM modules


class MultiProcessBatch:
    def __init__(self, uri, openclip_finetuned, llm_model, llm_processor, temperature):
        self.uri = uri
        self.openclip_finetuned = openclip_finetuned
        self.llm_model = llm_model
        self.llm_processor = llm_processor
        self.temperature = temperature
        self.names = None
        self.locationDetails = None
    # uuid4 id for vector database
    def generateId(self):
        return str(uuid.uuid4())

    # convert image date time to timestamp
    def timestamp(self, uri):
        ts = lu.getTimestamp(uri)
        return ts

    # get location details as: latitude, longitude and address
    def locationDetails(self, uri):
        lat_lon = lu.gpsInfo(uri)
        if lat_lon == ():
            lat_lon = (d_latitude, d_longitude)
        loc = lu.getLocationDetails(lat_lon)
        print(lat_lon, loc)
        return lat_lon[0], lat_lon[1], loc

    # get names of people in image
    def namesOfPeople(self, uri, openclip_finetuned):
        names = en.getEntityNames(uri, openclip_finetuned)
        return names

    # get image description from LLM
    def describeImage(self):
        d = LLM.fetch_llm_text(
            imUrl=self.uri,
            model=self.llm_model,
            processor=self.llm_processor,
            top=0.9,
            temperature=0.9,
            question="Answer with well organized thoughts, please describe the picture with insights.",
            people=self.names,
            location=dict.get("loc"))
        return d

    def llm_workflow(self):
        str_uuid = self.generateId()
        ts = self.timestamp(uri=self.uri) 
        self.location_details = self.locationDetails(uri=self.uri)
        self.names = self.namesOfPeople(self.uri, self.openclip_finetuned)
        text = self.describeImage()

def run_workflow(
    df,
    image_dir_path,
    chunk_size,
    metadata_path,
    metadata_file,
    number_of_instances,
    openclip_finetuned,
):
    m, t, p = LLM.setLLM()    
    
    progress_generation = st.sidebar.empty()
    bar = st.sidebar.progress(0)

    num = df.shape[0]
    with st.status("Generating LLM responses...", expanded=True) as status:
        img_iterator = mu.getRecursive(image_dir_path, chunk_size=chunk_size)
        count = 0
        for ilist in img_iterator:
            rlist = mu.is_processed_batch(ilist, df)
            if len(rlist) > 0:
                # asyncio.run(
                #     amain(
                #         ilist,
                #         metadata_path,
                #         metadata_file,
                #         number_of_instances,
                #         openclip_finetuned,
                #     )
                # )
                print(rlist)
            count = count + len(ilist)
            count = num if count > num else count
            progress_generation.text(
                f"{count} files processed out-of {num} => {int((100 / num) * count)}% processed"
            )
            bar.progress(int((100 / num) * count))
    status.update("process completed!", status="complete", extended=False)

def execute():
    (image_dir_path, metadata_path, metadata_file, chunk_size, number_of_instances, openclip_finetuned) = config.preprocess_config_load()

    st.sidebar.subheader("Metadata Grneration")
    st.sidebar.divider()
    df = None
    try:
        if os.path.exists(os.path.join(metadata_path, metadata_file)):
            data = []
            with open(os.path.join(metadata_path, metadata_file), mode="r") as f:
                res = f.read()
                res = res.replace("\n", "")
                res = res.replace("}{", "},{")
                res = "[" + res + "]"
                data = json.loads(res)
                df = pd.DataFrame(data)
                df = mu.drop_except(df, ["url"])
    except Exception as e:
        st.error(f"exception: {e} occured in loading metadata file")

    run_workflow(
        df,
        image_dir_path,
        chunk_size,
        metadata_path,
        metadata_file,
        number_of_instances,
        openclip_finetuned,
    )


# kick-off metadata generation
if __name__ == "__main__":
    execute()
