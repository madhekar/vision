import os
import glob
from PIL import Image
import pyiqa
import torch
from torchvision.transforms import ToTensor
from utils.config_util import config
from utils.util import model_util as mu
from utils.util import statusmsg_util as sm
from utils.util import storage_stat as ss

import asyncio
import multiprocessing as mp
import aiofiles
import aiomultiprocess as aiomp
from aiomultiprocess import Pool
from functools import partial
import streamlit as st

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = ToTensor()

def create_metric():
    print(pyiqa.list_models())
    iqa_metric = pyiqa.create_metric("niqe", device=device)
    return iqa_metric

iqa_metric = create_metric()

async def is_valid_size_and_score(args, img):
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
async def archive_images(image_path, archive_path, bad_quality_tuple_list):                
    if len(bad_quality_tuple_list) != 0:
        space_saved = 0
        image_cnt =0 
        for quality in bad_quality_tuple_list:
                space_saved += os.path.getsize(os.path.join(quality[0], quality[1]))
                image_cnt += 1
                #uuid_path = mu.create_uuid_from_string(quality[0]) 
                uuid_path = mu.extract_subpath(image_path, quality[0])
                if not os.path.exists(os.path.join(archive_path, uuid_path)):
                    os.makedirs(os.path.join(archive_path, uuid_path))
                os.rename( os.path.join(quality[0], quality[1]), os.path.join(archive_path, uuid_path, quality[1]))
                print(f"{quality} Moved Successfully!")
                #sm.add_messages("quality", f"s| file {quality} moved successfully.")

        print(f"\n\n saved {round(space_saved / 1000000)} mb of Space, {image_cnt} images archived.")
        sm.add_messages("quality",f"w| saved {round(space_saved / 1000000)} mb of Space, {image_cnt} images archived.")
    else:
        print("No bad quality images Found :)")
        sm.add_messages("quality", "w| no bad quality images Found.")


async def iq_work_flow(image_dir_path, archive_path, threshold):

    #lock = asyncio.Lock()
    chunk_size = int(mp.cpu_count())

    img_iterator = mu.getRecursive(image_dir_path, chunk_size=10)
    result = []
    #with st.status("Generating LLM responses...", expanded=True) as status:
    async with Pool(processes=chunk_size,  maxtasksperchild=1) as pool: 
        #count = 0
        res = [] 
        for il in img_iterator:
            if len(il) > 0:
                res = await asyncio.gather(
                        pool.map(partial(is_valid_size_and_score, threshold), il))
                result.append(res)
                #await archive_images(image_dir_path, archive_path, res)
    pool.close()

    archive_images(image_dir_path, archive_path, result)            
"""
    input_image_path,
    archive_quality_path,
    image_quality_threshold,

"""
def execute(source_name):

    aiomp.set_start_method('fork')
    (input_image_path, archive_quality_path,image_quality_threshold) = config.image_quality_config_load()

    input_image_path_updated = os.path.join(input_image_path,source_name)
    
    arc_folder_name = mu.get_foldername_by_datetime()  
    
    archive_quality_path = os.path.join(archive_quality_path, source_name, arc_folder_name)

    #iqa_metric = create_metric()

    asyncio.run(iq_work_flow(input_image_path_updated, archive_quality_path, image_quality_threshold))

    ss.remove_empty_files_and_folders(input_image_path_updated)
    
if __name__ == "__main__":
    execute(source_name="")