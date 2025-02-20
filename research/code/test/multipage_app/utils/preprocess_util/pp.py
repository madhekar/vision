import os
import uuid
from utils.preprocess_util import awaitUtil
from utils.preprocess_util import entities as en
from utils.preprocess_util import LLM
import json
import glob
import pandas as pd
import streamlit as st
from utils.config_util import config
from utils.util import location_util as lu
from utils.util import model_util as mu

from multiprocessing import Pool, freeze_support
from itertools import repeat

d_latitude, d_longitude = 32.96887205555556, -117.18414305555557


# init LLM modules

# uuid4 id for vector database
def generateId():
    return str(uuid.uuid4())

# convert image date time to timestamp
def timestamp(uri):
    ts = lu.getTimestamp(uri)
    return ts

# get location details as: latitude, longitude and address
def locationDetails(uri):
    lat_lon = lu.gpsInfo(uri)
    if lat_lon == ():
        lat_lon = (d_latitude, d_longitude)
    loc = lu.getLocationDetails(lat_lon)
    print(lat_lon, loc)
    return lat_lon[0], lat_lon[1], loc

# get names of people in image
def namesOfPeople(uri, openclip_finetuned):
    names = en.getEntityNames(uri, openclip_finetuned)
    return names

# get image description from LLM
def describeImage(uri, llm_model, llm_processor, names, location):
    d = LLM.fetch_llm_text(
        imUrl=uri,
        model=llm_model,
        processor=llm_processor,
        top=0.9,
        temperature=0.9,
        question="Answer with well organized thoughts, please describe the picture with insights.",
        people=names,
        location=location
    )
    return d

def llm_workflow(uri, lm, lp, openclip_finetuned):
    suuid = generateId()
    ts = timestamp(uri) 
    location_details = locationDetails(uri)
    names = namesOfPeople(uri, openclip_finetuned)
    text = describeImage(uri, lm, lp, names, location_details)
    #st.info(f'debug: {suuid} : {ts}' ) # : {location_details} : {names} : {text}')
    return (suuid, ts, location_details, names, text)

# recursive call to get all image filenames
def getRecursive(rootDir, chunk_size=10):
    f_list = []
    for fn in glob.iglob(rootDir + "/**/*", recursive=True):
        if not os.path.isdir(os.path.abspath(fn)):
            f_list.append(os.path.abspath(fn))
    for i in range(0, len(f_list), chunk_size):
        yield f_list[i : i + chunk_size]

def run_workflow(
    df,
    image_dir_path,
    chunk_size,
    metadata_path,
    metadata_file,
    number_of_instances,
    openclip_finetuned,
):
    freeze_support()
    m, t, p = LLM.setLLM()    
    
    progress_generation = st.sidebar.empty()
    bar = st.sidebar.progress(0)

    num = df.shape[0]
    #pool = mp.ProcessPool(4)  # Create a pool with 4 processes

    img_iterator = mu.getRecursive(
                "/home/madhekar/work/home-media-app/data/train-data/img",
                chunk_size=chunk_size,
            )
    
    with st.status("Generating LLM responses...", expanded=True) as status:

        with Pool() as pool:
            count = 0

            for ilist in img_iterator:
                rlist = mu.is_processed_batch(ilist, df)
                if len(rlist) > 0:
                    status.info(rlist)
                    ret = pool.starmap(llm_workflow, zip(rlist, repeat(m), repeat(p), repeat(openclip_finetuned)))
                    print(f":->{ret}:")
                count = count + len(ilist)
                count = num if count > num else count
                progress_generation.text(f"{count} files processed out-of {num} => {int((100 / num) * count)}% processed")
                bar.progress(int((100 / num) * count))
        #pool.close()
        #pool.join()

    status.update(label="process completed!", state="complete", expanded=False)

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
