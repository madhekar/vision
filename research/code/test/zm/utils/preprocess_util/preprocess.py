import os
import uuid
import logging
import itertools as it
import cv2
import numpy as np
from utils.preprocess_util import awaitUtil
#from utils.preprocess_util import entities as en
from utils.preprocess_util import LLM_Next, ollama_llava_next
import json
import glob
import pandas as pd
import streamlit as st
import tqdm
from utils.config_util import config
from utils.util import location_util as lu
from utils.util import model_util as mu
from utils.util import fast_parquet_util as fpu
from utils.util import storage_stat as ss
from utils.face_util import base_face_predict as bft
from utils.face_detection_util import face_predictor as fpr
from utils.filter_util import filter_inferance as fi
import tensorflow as tf
from tensorflow.keras.models import load_model

import asyncio
import multiprocessing as mp
import aiofiles
import aiomultiprocess as aiomp
from aiomultiprocess import Pool
from utils.util import ball_tree as bt
from functools import partial
#import dill as pickle

d_latitude, d_longitude = 32.968689, -117.184243
d_loc = 'Madhekar residence in Carmel Valley'
#m, t, p = LLM.setLLM()
#client = ollama_llava_next.create_default_client()
ap, fmodel, fle = fpr.init_predictor_module()
#fm, fc,isz = fi.init_filter_model()

p = LLM_Next.setLLM()
#btree = st.empty()
#ocfine = "/home/madhekar/work/home-media-app/models/zeshaOpenClip/clip_finetuned.pth"
#global_face = bft.base_face_res()
"""
exception: Traceback (most recent call last): File "/home/madhekar/work/vision/research/code/test/zm/.venv/lib/python3.10/site-packages/aiomultiprocess/pool.py", 
line 110, in run result = future.result() 
File "/home/madhekar/work/vision/research/code/test/zm/utils/preprocess_util/preprocess.py", 
line 90, in timestamp ts,uc = lu.getTimestamp(uri) #lu.get_image_exif_info(uri) File "/home/madhekar/work/vision/research/code/test/zm/utils/util/location_util.py", 

line 292, in getTimestamp return value, s_user_comment UnboundLocalError: local variable 's_user_comment' referenced before assignment occurred in async main function
-------

exception: Traceback (most recent call last): File "/home/madhekar/work/vision/research/code/test/zm/.venv/lib/python3.10/site-packages/aiomultiprocess/pool.py", line 110,
 in run result = future.result() File "/home/madhekar/work/vision/research/code/test/zm/utils/preprocess_util/preprocess.py", 
 line 158, in describeImage d = LLM_Next.fetch_llm_text(imUrl=uri, pipe=p, question="Describe the image with thoughtful insights using additional information provided. ", partial_prompt=ppt, location=location) File "/home/madhekar/work/vision/research/code/test/zm/utils/preprocess_util/LLM_Next.py", 
 line 79, in fetch_llm_text image = Image.open(imUrl).convert("RGB") File "/home/madhekar/work/vision/research/code/test/zm/.venv/lib/python3.10/site-packages/PIL/Image.py", 
 line 916, in convert self.load() File "/home/madhekar/work/vision/research/code/test/zm/.venv/lib/python3.10/site-packages/PIL/ImageFile.py", line 266, in load raise OSError(msg) OSError: image file is truncated (0 bytes not processed) occurred in async main function

"""
# init LLM modules
@st.cache_resource
def location_initialize(smp, smf):
    try:
        df = fpu.init_location_cache(os.path.join(smp, smf))
    except Exception as e:
        st.error(f"exception occurred in loading location metadata: {smf} with exception: {e}")  
    return df 

@st.cache_resource
def location_initialize_btree(smp, smf):
    try:
        btree_ = fpu.init_location_btree_cache(os.path.join(smp, smf))
    except Exception as e:
        st.error(f"exception occurred in loading location metadata: {smf} with exception: {e}")  
    return btree_ 
    
def get_loc_name_by_latlon(latlon):
    # if latlon:
    #     print(f'****search loc by latlon: {latlon}')

        #row = st.session_state.df_loc.loc[st.session_state.df_loc.LatLon == latlon].values.flatten().tolist()

    nm = st.session_state.ball_tree.get_location_name_for_latlong(latlon[0], latlon[1])
    print(f'latlon: {latlon} => loc name: {nm}')
    return nm

# uuid4 id for vector database
async def generateId(args):
    uri= args
    return str(uuid.uuid4()) #[uri, str(uuid.uuid4())]


# convert image date time to timestamp
"""
exception: Traceback (most recent call last): File "/home/madhekar/work/vision/research/code/test/zm/.venv/lib/python3.10/site-packages/aiomultiprocess/pool.py", line 110, in run result = future.result() 
File "/home/madhekar/work/vision/research/code/test/zm/utils/preprocess_util/preprocess.py", line 85, in timestamp ts,uc = lu.getTimestamp(uri) #lu.get_image_exif_info(uri) 
File "/home/madhekar/work/vision/research/code/test/zm/utils/util/location_util.py", line 246, in getTimestamp date_time = exifdata[36867] KeyError: 36867 occred in async main function
"""
async def timestamp(args):
    uri = args
    print(f'ts--{uri}')
    ts,uc = lu.getTimestamp(uri) #lu.get_image_exif_info(uri) 
    return str(ts), str(uc)

async def faces_partial_prompt(args):
    uri = args
    txt = fpr.predict_img_faces(ap, uri, fmodel, fle) 
    return txt

# async def img_type_detection(args, lock):
#     url = args

#     print(f"***{url}")
#     # itype = fi.predict_image(uri, fm, fc, isz)
#     img = tf.keras.preprocessing.image.load_img(url, target_size=isz)
#     img_array = tf.keras.preprocessing.image.img_to_array(img)
#     img_array = tf.expand_dims(img_array, 0)  # Create a batch
#     img_array = img_array / 255.0  # Rescale pixels

#     # img = Image.open(image_path)
#     # img = img.resize(image_size)
#     # img_array = np.array(img)
#     # img_array = np.expand_dims(img_array, 0)
#     # img_array = img_array / 255.0

#     print(f"---*&* {img_array}")
#     predictions = fm.predict(img_array)
#     predicted_class = fc[np.argmax(predictions)]
#     confidence = np.max(predictions)

#     print(f"Image: {url}")
#     print(f"Predicted class: {predicted_class} with confidence {confidence:.2f}")
#     return predicted_class


    # print(f'***{itype}')
    # return itype
   
# get location details as: (latitude, longitude) and address
async def locationDetails(args, lock):
    uri = args
    async with lock:
      try:  
        #print(f"-->> {uri}")
        loc = ""
        lat_lon = ()
        lat_lon = lu.gpsInfo(uri)
        print(f' {lat_lon} : {uri}')
        if lat_lon == ():
            lat_lon = (d_latitude, d_longitude) # default location lat, lon
            loc = d_loc # default location description
        else:    
           loc = get_loc_name_by_latlon(lat_lon)
        #print(f"--->> {lat_lon} : {loc}")
        if not loc:
            loc = lu.getLocationDetails(lat_lon, max_retires=3)
      except Exception as e:
        st.error(f'exception: {e} occurred in getting lat/ lon or location details for {uri}')
      return str(lat_lon), loc
     
# get image description from LLM
async def describeImage(args):
    ppt, location, uri = args
    ppt1 = ppt if ppt != None else "na"
    print(f'ppt: {ppt1}, location: {location}, url: {uri}')
    #d = LLM_Next.fetch_llm_text(imUrl=uri, pipe=p, question="Describe the image with thoughtful insights using additional information provided. ", partial_prompt=ppt, location=location)
    d =  await ollama_llava_next.describe_image( uri, ppt1, location)
    return d

"""
/home/madhekar/work/home-media-app/models/zeshaOpenClip/clip_finetuned.pth
"""
# recursive call to get all image filenames, to be replaced by parquet generator
def getRecursive(rootDir, chunk_size=10):
    f_list = []
    for fn in glob.iglob(rootDir + "/**/*", recursive=True):
        if not os.path.isdir(os.path.abspath(fn)):
            f_list.append(os.path.abspath(fn))
    for i in range(0, len(f_list), chunk_size):
        yield f_list[i : i + chunk_size]

def new_xform(res):
    ll = [list(x) for x in zip(*res)]
    lr = [[row[2], row[3][1], row[4]] for row in ll]
    lf = [[row[4], row[0],row[1][0], row[1][1],row[3][0], row[3][1], row[2]] for row in ll]
    return lr, lf

def final_xform(alist):
    keys = [ 'uri', 'id', 'ts','type','latlon' ,'loc', 'ppt', 'text']
    print(alist)
    return [{k:v for k,v in zip(keys, sublist)} for sublist in alist]

# appends json rows to file
async def append_file(filename, dict_data_list, mode):
    async with aiofiles.open(filename, mode) as f:
        for dict_element in dict_data_list:
           stv = json.dumps(dict_element)
           #info(stv)
           await f.write(stv)
           await f.write(os.linesep)
        await f.close()       

def setup_logging(level=logging.WARNING):
    logging.basicConfig(level=level)  

"""
multi processing linux tools
ps -ef | grep -w streamlit
pgrep --count streamlit
killall -9 streamlit
"""
async def run_workflow(
    df,
    image_dir_path,
    chunk_size,
    queue_size,
    metadata_path,
    metadata_file,
    number_of_instances,
    openclip_finetuned
):
    st.info(f"CPU COUNT: {chunk_size}")
    
    progress_generation = st.sidebar.empty()
    bar = st.sidebar.progress(0)
    if df is not None:
      if not df.empty:
        num = df.shape[0]
      else:
        num=0
    else:
      num = 0
     
    num_files = len(glob.glob(os.path.join(image_dir_path,'/**/*')))
    num = num_files - num
    print(f'processing files in: {image_dir_path} total files: {num_files}') 
    

    lock = asyncio.Lock()
    img_iterator = mu.getRecursive(image_dir_path, chunk_size=chunk_size)

    with st.status("Generating LLM responses...", expanded=True) as status:
        async with Pool(processes=chunk_size,  queuecount=queue_size, maxtasksperchild=1) as pool:  #initializer=pool_init, initargs=(bfs,),
            count = 0
            res = []
            for ilist in img_iterator:
                rlist = mu.is_processed_batch(ilist, df)
                print(rlist)
                if len(rlist) > 0:
                    res = await asyncio.gather(
                        pool.map(generateId, rlist),
                        pool.map(timestamp, rlist),
                        pool.map(faces_partial_prompt, rlist),
                        #pool.map(partial(img_type_detection,lock=lock), rlist),
                        pool.map(partial(locationDetails, lock=lock), rlist)
                    )
                    
                    res.append(rlist)

                    print('*===>', res)
                    rflist, oflist = new_xform(res)

                    res1 = await asyncio.gather(pool.map(describeImage,  rflist))
                    #res1 = await pool.map(describeImage,  rflist)
                    print('****', res1)

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
        #st.info(res)
        pool.close()
        # pool.join()

    status.update(label="process completed!", state="complete", expanded=False)


def execute(user_source_selected):
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

    static_metadata_path = os.path.join(static_metadata_path, user_source_selected)
    # add user data source to image input and metadata output paths
    image_dir_path = os.path.join(image_dir_path, user_source_selected)
    
    if not os.path.exists(image_dir_path):
        st.error(f'exception: image data path for {user_source_selected} does not exists!')
    metadata_path = os.path.join(metadata_path, user_source_selected)
    if not os.path.exists(metadata_path):
        os.makedirs(metadata_path)

    if "df_loc" not in st.session_state:
        df = location_initialize(static_metadata_path, static_metadata_file)
        st.session_state.df_loc = df
    else:
        df_loc = st.session_state.df_loc   

    if "ball_tree" not in st.session_state:
        btree = location_initialize_btree(static_metadata_path, static_metadata_file)
        st.session_state.ball_tree = btree
    else:
        ball_tree = st.session_state.ball_tree        

    #btree = location_initialize(static_metadata_path, static_metadata_file)    

    chunk_size = 1 #int(mp.cpu_count() // 8)
    queue_size = chunk_size
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
        st.error(f"exception: {e} occurred in loading metadata file")

    # bcreate_metadata = st.button("start metadata creation")
    # if bcreate_metadata:

    print(df)

    try:
        asyncio.run(run_workflow(
            df,
            image_dir_path,
            chunk_size,
            queue_size,
            metadata_path,
            metadata_file,
            number_of_instances,
            openclip_finetuned,
        ))
    except Exception as e:
        st.error(f'exception: {e} occurred in async main function')    


# kick-off metadata generation
if __name__ == "__main__":
    execute("user_source_selected")
