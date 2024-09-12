import os
import uuid
import asyncio
import util
import awaitUtil
import entities
import LLM
import aiofiles
import json

# init LLM modules
m, t, p = LLM.setLLM()

# queue processing rutine
async def worker( name, queue):
    async with aiofiles.open("data.json", mode="a") as f:
        print(f"{name}!")
        while True:
            # Get a "work item" out of the queue.
            dict = await queue.get()

            # Sleep for the "sleep_for" seconds.
            llmStr = await awaitUtil.force_awaitable(describeImage)(dict)

            dict["text"] = llmStr

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
        ts = util.getTimestamp(uri)
        return ts   

# get location details as: latitude, longitude and address    
def locationDetails( uri):
        lat_lon = util.gpsInfo(uri)
        loc = util.getLocationDetails(lat_lon)
        return lat_lon[0], lat_lon[1], loc 
    
# get names of people in image    
def namesOfPeople(uri):
        names = entities.getEntityNames(uri)
        return names

# get image description from LLM
def describeImage( dict):
        d = LLM.fetch_llm_text(
                imUrl=dict.get("url"),
                model=m,
                processor=p,
                top=0.9,
                temperature=0.9,
                question="Answer with organized thoughts: Please describe the picture, ",
                people=dict.get("names"),
                location=dict.get("loc")
            )
        return d 

# collect metadata for all images
async def make_request(url: str, semaphore: asyncio.Semaphore):
    async with semaphore:
        s1 = await awaitUtil.force_awaitable(generateId)(url)

        s2 = await awaitUtil.force_awaitable(timestamp)(url)
        
        await asyncio.sleep(1)

        s3 = await awaitUtil.force_awaitable(locationDetails)(url)

        r4 = await awaitUtil.force_awaitable(namesOfPeople)(url)

        return {"url" : url, "id": s1, "timestamp": s2, "lat": s3[0], "lon" : s3[1], "loc": s3[2], "names": r4}


# main asynchronous function 
async def amain(iList):
    queue = asyncio.Queue()  
    
    semaphore = asyncio.Semaphore(10)    

    tasks = [make_request(img_path ,semaphore) for img_path in iList]
    
    for co in asyncio.as_completed(tasks):
        res = await co
        print(res)
        await asyncio.sleep(1)
        queue.put_nowait(res)
        
    ts = []
    for i in range(10):
         t = asyncio.create_task(worker(f"worker-{i}", queue=queue))    
         ts.append(t)
    
    await queue.join()

    for t in ts:
         t.cancel()     

# kick-off metadata generation 
if __name__ == "__main__":
    img_iterator = util.getRecursive("/home/madhekar/Pictures", chunk_size=10)
    for lst in img_iterator:
       asyncio.run(amain(lst))