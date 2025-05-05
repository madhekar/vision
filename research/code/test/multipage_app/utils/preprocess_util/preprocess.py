import os
import uuid
import logging
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
from utils.util import fast_parquet_util as fpu
from utils.util import storage_stat as ss
from utils.face_util import base_face_test as bft

import asyncio
import multiprocessing as mp
import aiofiles
import aiomultiprocess as aiomp
from aiomultiprocess import Pool
from functools import partial
import dill as pickle

d_latitude, d_longitude = 32.968700, -117.184196
d_loc = 'madhekar residence at carmel vally san diego, california'
m, t, p = LLM.setLLM()
ocfine = "/home/madhekar/work/home-media-app/models/zeshaOpenClip/clip_finetuned.pth"

face_detect = bft.init()

# init LLM modules
@st.cache_resource
def location_initialize(smp, smf):
    try:
        df = fpu.init_location_cache(os.path.join(smp, smf))
    except Exception as e:
        st.error(f"exception occured in loading location metadata: {smf} with exception: {e}")  
    return df 

def get_loc_name_by_latlon(latlon):
    if latlon:
        row = st.session_state.df_loc.loc[st.session_state.df_loc.LatLon == latlon].values.flatten().tolist()
        if len(row) > 0:
           print(f'--> found location in cache: {row}')
           return row[0]
        else:
           return None

# uuid4 id for vector database
async def generateId(uri):
    return (uri, str(uuid.uuid4()))


# convert image date time to timestamp
async def timestamp(uri):
    print(f'ts--{uri}')
    ts = lu.getTimestamp(uri)
    return str(ts)


# get location details as: (latitude, longitude) and address
async def locationDetails(uri, lock):
    async with lock:
      try:  
        print(f"-->> {uri}")
        loc = ""
        lat_lon = ()
        lat_lon = lu.gpsInfo(uri)
        if lat_lon == ():
            lat_lon = (d_latitude, d_longitude) # default location lat, lon
            loc = d_loc # default location description
        else:    
           loc = get_loc_name_by_latlon(lat_lon)
        print(f"--->> {lat_lon} : {loc}")
        if not loc:
            loc = lu.getLocationDetails(lat_lon, max_retires=3)
      except Exception as e:
        st.error(f'exception occured in getting lat/ lon or location details for {uri}')
      return str(lat_lon), loc
     
# get names of people in image
async def namesOfPeople(uri, lock):
    async with lock:
      names =  en.getEntityNames(uri, ocfine)
      return names

async def facesNames(uri):
    print(uri)
    names = bft.predict_names(face_detect, uri)   
    print(names)
    return names

# get image description from LLM
async def describeImage(args):
    names, uri, location = args
    print(args)
    d =  LLM.fetch_llm_text(
        imUrl=uri,
        model=m,
        processor=p,
        top=0.9,
        temperature=0.9,
        question="Answer with well organized thoughts, please describe the picture with insights.",
        people=names,
        location=location,
    )
    print(d)
    return d

"""
/home/madhekar/work/home-media-app/models/zeshaOpenClip/clip_finetuned.pth
"""
# async def llm_workflow(uri):
#     #m, t, p = LLM.setLLM()
#     semaphore = asyncio.Semaphore(1)
#     suuid = await generateId(uri)
#     ts = await timestamp(uri)
#     location_details =  await locationDetails(uri, semaphore)
#     names =  await namesOfPeople(uri)
#     text =  await describeImage(uri, m, p, names, location_details)
#     return (uri, suuid, ts, location_details, names, text)


# recursive call to get all image filenames
def getRecursive(rootDir, chunk_size=10):
    f_list = []
    for fn in glob.iglob(rootDir + "/**/*", recursive=True):
        if not os.path.isdir(os.path.abspath(fn)):
            f_list.append(os.path.abspath(fn))
    for i in range(0, len(f_list), chunk_size):
        yield f_list[i : i + chunk_size]


def xform(res):
    fr=[]
    for k in range(len(res[0])):
      lr = [i[k] for i in res]
      fr.append(lr)

    df = pd.DataFrame(fr, columns=['url', 'ts', 'names', 'location'])   
    df[['uri', 'id']] = pd.DataFrame(df['url'].tolist(), index=df.index)
    df[['latlon','loc']] = pd.DataFrame(df['location'].tolist(), index=df.index)
    dfo= df.drop(columns=['url', 'location'])
    df.drop(columns=['url', 'ts', 'id', 'latlon', 'location'], inplace=True)
    #print(df.head())  
    lst = df.to_numpy().tolist() 
    print(lst)  
    return [tuple(e) for e in lst], dfo.to_numpy().tolist()

def final_xform(alist):
    keys = ['ts', 'names', 'uri', 'id', 'latlon', 'loc', 'text']
    return [{k:v for k,v in zip(keys, sublist)} for sublist in alist]


async def append_file(filename, dict_data_list, mode):
    async with aiofiles.open(filename, mode) as f:
        for dict_element in dict_data_list:
           stv = json.dumps(dict_element)
           st.info(stv)
           await f.write(stv)
           await f.write(os.linesep)
        await f.close()       

def setup_logging(level=logging.WARNING):
    logging.basicConfig(level=level)
"""
ps -ef | grep -w streamlit
pgrep --count streamlit
killall -9 streamlit
"""
async def run_workflow(
    df,
    image_dir_path,
    chunk_size,
    metadata_path,
    metadata_file,
    number_of_instances,
    openclip_finetuned,
):
    st.info(f"CPU COUNT: {chunk_size}")
    print(f"CPU COUNT: {chunk_size}")
    progress_generation = st.sidebar.empty()
    bar = st.sidebar.progress(0)
    if df is not None:
      if not df.empty:
        num = df.shape[0]
      else:
          num=0
    else:
      num = 0
    num_files = len(glob.glob(os.path.join(image_dir_path,'*')))
    num = num_files - num
    #semaphore = asyncio.Semaphore(1)

    lock = asyncio.Lock()

    img_iterator = mu.getRecursive(image_dir_path, chunk_size=chunk_size)

    with st.status("Generating LLM responses...", expanded=True) as status:
        async with Pool(processes=chunk_size, initializer=setup_logging, initargs=(logging.WARNING,), maxtasksperchild=1) as pool:
            count = 0
            res = []
            for ilist in img_iterator:
                rlist = mu.is_processed_batch(ilist, df)
                if len(rlist) > 0:

                    # res=[]
                    # fetch_result = [asyncio.create_task(llm_workflow(uri=u)) for u in rlist ]
                    # for ul in asyncio.as_completed(fetch_result):
                    #     res.extend(await(ul))

                    # async for ur in pool.map(llm_workflow, rlist): 
                    #     st.info(ur)
                    #     res.extend(ur)
                       
                    res = await asyncio.gather(
                        pool.map(generateId, rlist),
                        pool.map(timestamp, rlist),
                        #pool.map(namesOfPeople, rlist),
                        pool.map(partial(facesNames, lock=lock), rlist),
                        pool.map(partial(locationDetails, lock=lock), rlist)
                    )

                    #st.info(res)

                    rflist, oflist = xform(res)

                    #st.info(oflist)

                    res1 = await asyncio.gather(
                        pool.map(describeImage,  rflist)
                    )

                    #st.info(res1)

                    zlist = [oflist[i] + [res1[0][i]]  for i in range(len(oflist))]

                    fdictlist = final_xform(zlist)

                    st.info(fdictlist)

                    await append_file(os.path.join(metadata_path, metadata_file), fdictlist, 'a+')

                count = count + len(ilist)
                count = num if count > num else count
                if num > 0:
                    progress_generation.text(f"{count} files processed out-of {num} => {int((100 / num) * count)}% processed")
                    bar.progress(int((100 / num) * count))
                else:
                    progress_generation.text(f"{count} files processed out-of {num} => {int((100 / 1) * count)}% processed (all done!)")
                    bar.progress(int((100 / 1) * count))    
        st.info(res)
        pool.close()
        # pool.join()

    status.update(label="process completed!", state="complete", expanded=False)


def execute():
    #mp.freeze_support()
    aiomp.set_start_method("fork")

    (
        image_dir_path,
        metadata_path,
        metadata_file,
        chunk_size,
        number_of_instances,
        openclip_finetuned,
        static_metadata_path,
        static_metadata_file
    ) = config.preprocess_config_load()

    #image_dir_path = "/home/madhekar/work/home-media-app/data/train-data/img"    
    st.sidebar.subheader("User Storage Source", divider="gray")

    user_source_selected = st.sidebar.selectbox(
        "data source folder",
        options=ss.extract_user_raw_data_folders(image_dir_path),
        label_visibility="collapsed"
    )

    static_metadata_path = os.path.join(static_metadata_path, user_source_selected)
    # add user data source to image input and metadata output paths
    image_dir_path = os.path.join(image_dir_path, user_source_selected)
    
    if not os.path.exists(image_dir_path):
        st.error(f'excetion: image data path for {user_source_selected} does not exixts!')
    metadata_path = os.path.join(metadata_path, user_source_selected)
    if not os.path.exists(metadata_path):
        os.makedirs(metadata_path)

    if "df_loc" not in st.session_state:
        df = location_initialize(static_metadata_path, static_metadata_file)
        st.session_state.df_loc = df
    else:
        df_loc = st.session_state.df_loc   

    chunk_size = int(mp.cpu_count() // 2)
    st.sidebar.subheader("Metadata Generation")
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
                df = mu.drop_except(df, ["uri"])
    except Exception as e:
        st.error(f"exception: {e} occured in loading metadata file")

    # bcreate_metadata = st.button("start metadata creation")
    # if bcreate_metadata:

    asyncio.run(run_workflow(
        df,
        image_dir_path,
        chunk_size,
        metadata_path,
        metadata_file,
        number_of_instances,
        openclip_finetuned,
    ))


# kick-off metadata generation
if __name__ == "__main__":
    execute()
