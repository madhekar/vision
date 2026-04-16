import os
import time
from tqdm import tqdm
from utils.config_util import config
from utils.util import model_util as mu
from utils.util import statusmsg_util as sm
import multiprocessing as mp
import streamlit as st
from utils.util import video_util as vu

def check_video_quality(v):
    vu.crop_detect_and_crop_video_workaround(v)

def vq_work_flow(video_dir_path, chunk_size):

    nfiles = len(mu.getFiles(video_dir_path))
    print(f'---> nfiles: {nfiles}')
    vid_iterator = mu.getRecursive(video_dir_path,  chunk_size)
    result = []
    with tqdm(total=nfiles, desc='detecting poor quality files', unit='items', unit_scale=True) as pbar:
        #async with Pool(processes=chunk_size) as pool:
            res=[]
            for il in vid_iterator:
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

        input_video_path_updated = os.path.join(input_video_path, source_name)
        arc_folder_name = mu.get_foldername_by_datetime()     
        archive_quality_path = os.path.join(archive_quality_path, source_name, arc_folder_name)

        chunk_size = int(mp.cpu_count())
        # queue_count = chunk_size

        sm.add_messages("quality", f"s| number of video parallel processes {chunk_size}")

        start = time.time()
        try:
            vq_work_flow(
                input_video_path_updated,
                chunk_size
            )
        except Exception as e:
            st.error(f'exception: {e} occurred in execute function') 
        processing_duration = int(time.time() - start)
        print(f"processing duration: {processing_duration} seconds")
        sm.add_messages("quality", f"s| processing duration: {processing_duration} seconds")

    except Exception as e:
        sm.add_messages("quality", f"e| Exception occurred {e}")
        result = "failed"
    return result 