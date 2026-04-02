import os
import time
import subprocess
import shlex
import pandas as pd
from PIL import Image
import codecs
import pyiqa
import torch
from tqdm import tqdm
from torchvision.transforms import ToTensor
from utils.config_util import config
from utils.util import model_util as mu
from utils.util import statusmsg_util as sm
from utils.util import storage_stat as ss
from utils.util import location_util as lu
#from utils.filter_util import filter_inferance as fi
from utils.filter_util import filter_torch_inference as fti

from ast import literal_eval
import asyncio
import multiprocessing as mp
from multiprocessing import Pool
#import aiofiles
#import aiomultiprocess as aiomp
#from aiomultiprocess import Pool
from functools import partial
import streamlit as st
from utils.util import video_util as vu

def check_video_quality(v):
    vu.corp_detect_and_crop_video(v)

def vq_work_flow(video_dir_path, archive_path, threshold, chunk_size, filter_list):
    print(f"%%% {filter_list}")
    str_filter = ' '.join(filter_list)
    nfiles = len(mu.getFiles(video_dir_path))
    print(f'---> nfiles: {nfiles}')
    img_iterator = mu.getRecursive(video_dir_path,  chunk_size)
    result = []
    with tqdm(total=nfiles, desc='detecting poor quality files', unit='items', unit_scale=True) as pbar:
        #async with Pool(processes=chunk_size) as pool:
            res=[]
            for il in img_iterator:
                  if len(il) > 0:
                    fres = list(map(check_video_quality, il))
                    print("***", fres)
                    print(f'combined (quality & filtered): {res} len of batch: {len(il)}')
                    pbar.update(len(il))
        #pool.close()
        #pool.join()
    print(result)



def execute(source_name, filter_list):
    result = "success"
    try:
        print(filter_list)
        #mp.set_start_method("fork")
        #mp.freeze_support()
        (input_image_path, input_video_path, archive_quality_path, image_quality_threshold) = config.image_quality_config_load()

        input_image_path_updated = os.path.join(input_image_path, source_name)
        input_video_path_updated = os.path.join(input_video_path, source_name)
        arc_folder_name = mu.get_foldername_by_datetime()     
        archive_quality_path = os.path.join(archive_quality_path, source_name, arc_folder_name)

        # chunk_size = int(mp.cpu_count())
        # queue_count = chunk_size

        # sm.add_messages("quality", f"s| number of parallel processes {chunk_size}")

        start = time.time()
        try:
            # iq_work_flow(
            #     input_video_path_updated,
            #     archive_quality_path,
            #     image_quality_threshold,
            #     chunk_size,
            #     filter_list
            # )
        except Exception as e:
            st.error(f'exception: {e} occurred in execute function') 
        processing_duration = int(time.time() - start)
        print(f"processing duration: {processing_duration} seconds")
        sm.add_messages("quality", f"s| processing duration: {processing_duration} seconds")

        ss.remove_empty_image_files_and_folders(input_image_path_updated)

    except Exception as e:
        sm.add_messages("quality", f"e| Exception occurred {e}")
        result = "failed"
    return result 