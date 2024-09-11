import asyncio
import os
from random import randint


async def get_text(f):
    wait_time = randint(1, 3)
    print("downloading {} will take {} second(s)".format(f, wait_time))
    await asyncio.sleep(wait_time)  # I/O, context will switch to main function
    print("downloaded {}".format(f))


sem = asyncio.Semaphore(3)


async def safe_get_llm_text(f):
    async with sem:  # semaphore limits num of simultaneous downloads
        return await get_text(f)


async def main():
    tasks = [ asyncio.ensure_future(safe_get_llm_text(f))  # creating task starts coroutine
        for f in os.listdir(".")
    ]
    await asyncio.gather(*tasks)  # await moment all downloads done


if __name__ ==  "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
