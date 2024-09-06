import os
import uuid
import asyncio
import time
import util
import awaitUtil


from random import randint

def generateId(self):
    return str(uuid.uuid4())


def timestamp( uri):
        ts = util.getTimestamp(uri)
        return ts   
    
def locationDetails( uri):
        lat_lon = util.gpsInfo(uri)
        loc = util.getLocationDetails(lat_lon)
        return lat_lon[0], lat_lon[1], loc 
    
async def namesOfPeople(uri):
        #async with sem:
        wait_time = randint(3, 7)
        await asyncio.sleep(wait_time)
        return f"namesOfPeople done."

async def describeImage(uri):
        #async with sem:
        wait_time = randint(5, 10)
        await asyncio.sleep(wait_time)
        return f"describeImage done." 

async def make_request(url: str, semaphore: asyncio.Semaphore):
    async with semaphore:
        s1 = await awaitUtil.force_awaitable(generateId)(url)

        s2 = await awaitUtil.force_awaitable(timestamp)(url)

        s3 = await awaitUtil.force_awaitable(locationDetails)(url)

        r4 = await namesOfPeople(url)

        r5 = await describeImage(url)

    return (s1, s2, s3, r4, r5)


async def amain():
    semaphore = asyncio.Semaphore(2)
    imgs_path = util.getRecursive("/home/madhekar/Pictures")
    tasks = [make_request(img_path ,semaphore) for img_path in imgs_path]
    for cor in asyncio.as_completed(tasks):
        res = await cor
        print(res)


if __name__ == '__main__':
    asyncio.run(amain())