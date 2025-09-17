import os
import time
from PIL import Image
import pyiqa
import torch
from stqdm import stqdm
from torchvision.transforms import ToTensor
from utils.config_util import config
from utils.util import model_util as mu
from utils.util import statusmsg_util as sm
from utils.util import storage_stat as ss

import asyncio
import multiprocessing as mp
from multiprocessing import Pool
#import aiofiles
#import aiomultiprocess as aiomp
#from aiomultiprocess import Pool
from functools import partial
import streamlit as st

"""
madhekar@madhekar-UM690:~/work/vision/research/code/test/multipage_app$ inxi -G
Graphics:
  Device-1: AMD Rembrandt driver: amdgpu v: kernel
  Display: x11 server: X.Org v: 1.21.1.4 driver: X: loaded: amdgpu,ati
    unloaded: fbdev,modesetting,vesa gpu: amdgpu resolution: 3840x2160~30Hz
  OpenGL:
    renderer: REMBRANDT (rembrandt LLVM 15.0.7 DRM 3.42 5.15.0-144-generic)
    v: 4.6 Mesa 23.2.1-1ubuntu3.1~22.04.3

     pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.4/
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = ToTensor()

def create_metric():
    print(pyiqa.list_models())
    iqa_metric = pyiqa.create_metric("niqe", device=device)
    return iqa_metric

iqa_metric = create_metric()

def is_valid_size_and_score(args, img):
      threshold = args
      try:  
        if img is not None and os.path.getsize(img) > 512:
            im = Image.open(img).convert("RGB")
            h, w = im.size

            if w < 512 or h < 512:
                return img
            
            im_tensor = transform(im).unsqueeze(0).to(device)
            score = iqa_metric(im_tensor)
            f_score = score.item()

            print(f'{img} :: {h}:{w} :: {f_score}')

            res = img if f_score > threshold else ""
            return res
        else:
            sm.add_messages('quality', 'e| unable to load - NULL / Invalid image')
      except Exception as e:
                sm.add_messages("quality", f"e| error: {e} occurred while opening the image: {os.path.join(img[0], img[1])}")
      return img    

"""
Archive quality images
"""
def archive_images(image_path, archive_path, bad_quality_path_list):                
    if len(bad_quality_path_list) != 0:
        space_saved = 0
        image_cnt =0 
        for quality in stqdm(bad_quality_path_list, total=len(bad_quality_path_list)):
                space_saved += os.path.getsize(os.path.join(quality))
                image_cnt += 1
                #uuid_path = mu.create_uuid_from_string(quality[0]) 
                uuid_path = mu.extract_subpath(image_path, os.path.dirname(quality))
                if not os.path.exists(os.path.join(archive_path, uuid_path)):
                    os.makedirs(os.path.join(archive_path, uuid_path))
                os.rename( quality, os.path.join(archive_path, uuid_path, os.path.basename(quality)))
                print(f"{quality} Moved Successfully!")
                sm.add_messages("quality", f"s| file {quality} moved successfully.")

        print(f"\n\n saved {round(space_saved / 1000000)} mb of Space, {image_cnt} images archived.")
        sm.add_messages("quality",f"s| saved {round(space_saved / 1000000)} mb of Space, {image_cnt} images archived.")
    else:
        print("No bad quality images Found :)")
        sm.add_messages("quality", "s| no bad quality images Found.")


def iq_work_flow(image_dir_path, archive_path, threshold, chunk_size, queue_count):
    stqdm_container = st.container()
    nfiles = len(mu.getFiles(image_dir_path))
    img_iterator = mu.getRecursive(image_dir_path,  chunk_size)
    pbar = stqdm(total=nfiles, unit='files', unit_scale=True, unit_divisor=chunk_size)
    result = []
    with stqdm_container:
        with Pool(processes=chunk_size) as pool:
            res=[]
            for il in img_iterator:
                  if len(il) > 0:
                     pool.map(partial(is_valid_size_and_score, threshold),il)
                     result.append(res)
                     pbar.update(len(il))
        pool.close()
        pool.join()

    archive_images( image_dir_path, archive_path, [e for sb1 in result for sb2 in sb1 for e in sb2 if not e == ""])

def execute(source_name):
    print('----> in execute')
    result = "success"
    try:
        #mp.set_start_method("fork")
        mp.freeze_support()
        (input_image_path, archive_quality_path, image_quality_threshold) = config.image_quality_config_load()

        input_image_path_updated = os.path.join(input_image_path, source_name)
        arc_folder_name = mu.get_foldername_by_datetime()     
        archive_quality_path = os.path.join(archive_quality_path, source_name, arc_folder_name)
        # sm.info(f's| input images path: {input_image_path} out archive path: {archive_quality_path}')

        chunk_size = int(mp.cpu_count()) // 2
        queue_count = chunk_size

        sm.add_messages("quality", f"s| processes {chunk_size}")
        print(f"processes: {chunk_size} queue count: {queue_count}")

        start = time.time()
        iq_work_flow(
            input_image_path_updated,
            archive_quality_path,
            image_quality_threshold,
            chunk_size,
            queue_count,
        )
        processing_duration = int(time.time() - start)
        print(f"processing duration: {processing_duration}")
        sm.add_messages("quality", f"s| processing duration: {processing_duration}.")

        ss.remove_empty_files_and_folders(input_image_path_updated)

    except Exception as e:
        sm.add_messages("quality", f"e| Exception occurred {e}")
        result = "failed"
    print('----result--->', result)
    return result 

    
    
if __name__ == "__main__":
    execute(source_name="madhekar")

