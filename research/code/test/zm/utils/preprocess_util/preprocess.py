import os
import uuid
import logging
import itertools as it
from utils.preprocess_util import awaitUtil
#from utils.preprocess_util import entities as en
from utils.preprocess_util import LLM_Next
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
ap, fmodel, fle = fpr.init_predictor_module()
p = LLM_Next.setLLM()
#btree = st.empty()
#ocfine = "/home/madhekar/work/home-media-app/models/zeshaOpenClip/clip_finetuned.pth"
#global_face = bft.base_face_res()
"""

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
async def timestamp(args):
    uri = args
    #print(f'ts--{uri}')
    ts = lu.getTimestamp(uri)
    return str(ts)

async def faces_partial_prompt(args):
    uri = args
    txt = fpr.predict_img_faces(ap, uri, fmodel, fle) 
    return txt
   
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
    uri, location, names, attrib = args
    print(args)
    d= LLM_Next.fetch_llm_text(imUrl=uri, pipe=p, question="Please take time to describe the picture with thoughtful insights ", people=names, attrib=attrib, location=location)
    # d =  LLM_Next.fetch_llm_text(
    #     imUrl=uri,
    #     model=m,
    #     processor=p,
    #     top=0.9,
    #     temperature=0.95,
    #     question="Please take time to describe the picture with thoughtful insights ",
    #     people=names,
    #     attrib = attrib,
    #     location=location,
    # )
    # st.info(d)
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
    print('+-+->', ll)
    lr = [list(it.chain(*item)) for item in ll]
    print('+++>', lr)
    df = pd.DataFrame(lr, columns=["uri", "id", "ts", "latlon", "loc", "url2", "names", "attrib"])
    print(df.head(5))
    df1 = df.drop(columns=["url2", "id", "ts", "latlon"], axis=1)
    dfo = df.drop(columns=["url2"], axis=1)
    lst = df1.to_numpy().tolist()
    #print('...>', lst)
    return [tuple(e) for e in lst], dfo.to_numpy().tolist()

def final_xform(alist):
    #print('--->', alist)
    keys = [ 'uri', 'id', 'ts','latlon', 'loc', 'names', 'attrib', 'text']
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
                    # prep names and emotion
                    # df_rl = pd.DataFrame(rlist, columns=['uri'])
                    # df_r = bft.exec_process(df_rl)
                    # rlist = df_r.values.tolist()
                    # print(rlist)

                    res = await asyncio.gather(
                        pool.map(generateId, rlist),
                        pool.map(timestamp, rlist),
                        pool.map(faces_partial_prompt, rlist),
                        pool.map(partial(locationDetails, lock=lock), rlist)
                    )
                    
                    res.append(rlist)

                    print('===>', res)
                    rflist, oflist = new_xform(res)

                    res1 = await asyncio.gather(pool.map(describeImage,  rflist))

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
        st.error(f'exception: image data path for {user_source_selected} does not exixts!')
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

    chunk_size = int(mp.cpu_count() // 4)
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
        st.error(f"exception: {e} occured in loading metadata file")

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
        st.error(f'exception: {e} occred in async main function')    


# kick-off metadata generation
if __name__ == "__main__":
    execute("user_source_selected")
