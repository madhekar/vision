import os
import uuid
import asyncio
import time
import util

from random import randint

async def generateId(self):
    return str(uuid.uuid4())


async def timestamp( uri):
        #async with sem:
        wait_time = randint(1, 3)
        await asyncio.sleep(wait_time)
        ts = util.getTimestamp()
        return ts   
    
async def locationDetails( uri):
        #async with sem:
        wait_time = randint(1, 3)
        await asyncio.sleep(wait_time)
        return f"locationDetails done."   
    
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
    """simulates request"""
    async with semaphore:
        s1 = await generateId(url)

        s2 = await timestamp(url)

        s3 = await locationDetails(url)

        r4 = await namesOfPeople(url)

        r5 = await describeImage(url)

    return (s1, s2, s3, r4, r5)


async def amain():
    """main wrapper."""
    semaphore = asyncio.Semaphore(2)
    tasks = [make_request(img ,semaphore) for img in sorted(os.listdir('/Users/emadhekar/erase_me/images/'))]
    for cor in asyncio.as_completed(tasks):
        res = await cor
        print(res)


if __name__ == '__main__':
    asyncio.run(amain())