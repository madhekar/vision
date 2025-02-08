import os
import uuid
import asyncio
import awaitUtil
import entities
import LLM
import aiofiles
import json
import streamlit as st
from utils.config_util import config
from utils.util import location_util as lu
from utils.util import model_util as mu

d_latitude, d_longitude = 32.96887205555556, -117.18414305555557


# init LLM modules
m, t, p = LLM.setLLM()

# queue processing rutine
async def worker( name, mp, mf, queue):
    async with aiofiles.open(mp + mf, mode="a") as f:
        print(f"{name}!")
        while True:
            # Get a "work item" out of the queue.
            dict = await queue.get()

            # Sleep for the "sleep_for" seconds.
            llmStr = await awaitUtil.force_awaitable(describeImage)(dict)

            dict["txt"] = llmStr

            st = json.dumps(dict)
            await f.write(st)
            await f.write(os.linesep)

            # Notify the queue that the "work item" has been processed.
            queue.task_done()

# uuid4 id for vector database
def generateId(self):
    return str(uuid.uuid4())

# convert image date time to timestamp
def timestamp( uri):
        ts = lu.getTimestamp(uri)
        return ts   

# get location details as: latitude, longitude and address    
def locationDetails( uri):
        lat_lon = lu.gpsInfo(uri)
        if lat_lon == ():
            lat_lon = (d_latitude, d_longitude)
        loc = lu.getLocationDetails(lat_lon)
        print(lat_lon, loc)
        return lat_lon[0], lat_lon[1], loc 
    
# get names of people in image    
def namesOfPeople(uri, openclip_finetuned):
        names = entities.getEntityNames(uri, openclip_finetuned)
        return names

# get image description from LLM
def describeImage(dict):
        d = LLM.fetch_llm_text(
                imUrl=dict.get("url"),
                model=m,
                processor=p,
                top=0.9,
                temperature=0.9,
                question="Answer with well organized thoughts, please describe the picture with insights.",
                people=dict.get("nam"),
                location=dict.get("loc")
            )
        st.info(f"describe Image: LLM text for: {dict.get('url')} is: {d}")
       
        return d 

# collect metadata for all images
async def make_request(url: str, openclip_finetuned: str, semaphore: asyncio.Semaphore):
    print(url)
    async with semaphore:
        s1 = await awaitUtil.force_awaitable(generateId)(url)

        s2 = await awaitUtil.force_awaitable(timestamp)(url)
        
        await asyncio.sleep(1)

        s3 = await awaitUtil.force_awaitable(locationDetails)(url)

        r4 = await awaitUtil.force_awaitable(namesOfPeople)(url, openclip_finetuned)

        st.info(f"make_request: {url} dt: {s2} lat: {s3[0]} lon:  {s3[1]} loc: {s3[2]} nam: {r4}")

        return {"url" : url, "id": s1, "timestamp": s2, "lat": s3[0], "lon" : s3[1], "loc": s3[2], "nam": r4}


# main asynchronous function 
async def amain(iList, metadata_path, metadata_file, chunk_size, openclip_finetuned):
    
    st.info(f'now processing batch of {chunk_size}')
    st.info(iList)

    queue = asyncio.Queue()  
    semaphore = asyncio.Semaphore(10)    

    tasks = [make_request(img_path , openclip_finetuned, semaphore) for img_path in iList]
    
    for co in asyncio.as_completed(tasks):
        res = await co
        await asyncio.sleep(1)
        queue.put_nowait(res)
        
    ts = []
    for i in range(chunk_size):
        t = asyncio.create_task(worker(f"worker-{i}", metadata_path, metadata_file, queue=queue))
        ts.append(t)

    await queue.join()

    for t in ts:
         t.cancel()     

def execute():
    (image_dir_path,
    metadata_path,
    metadata_file,
    chunk_size,
    number_of_instances,
    openclip_finetuned) = config.preprocess_config_load()
    with st.status("Generating LLM responses...", expanded=True) as status:

        st.info(f'processing images in {image_dir_path} in chunks of: {chunk_size}')

        img_iterator = mu.getRecursive(image_dir_path, chunk_size=chunk_size)

        for ilist in img_iterator:
                st.info(f'now creating async with {number_of_instances} to process metadata in {metadata_path} for LLM ')
                asyncio.run(amain(ilist, metadata_path, metadata_file, number_of_instances, openclip_finetuned))
        status.update("process completed!", status="complete", extended = False)
# kick-off metadata generation 
if __name__ == "__main__":
    execute()