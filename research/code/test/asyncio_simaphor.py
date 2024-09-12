import asyncio
import os
from random import randint

#from aiodecorators import Semaphore

semaphore = asyncio.Semaphore(2)

async def get_text(f):
    wait_time = randint(1, 3)
    print("get_text {} will take {} second(s)".format(f, wait_time))
    await asyncio.sleep(wait_time)  # I/O, context will switch to main function
    s = "get_text {}".format(f)
    return s

async def get_id(f):
    wait_time = randint(1, 3)
    print("get_id {} will take {} second(s)".format(f, wait_time))
    await asyncio.sleep(wait_time)  # I/O, context will switch to main function
    s = "get_id {}".format(f)
    return s

#@Semaphore(2)
async def test(i):
  async with semaphore:  
    print(f"Start Task {i}")
    await asyncio.sleep(2)
    print(f"Finished Task {i}")
    s1 = await get_text(i)
    s2 = await get_id(i)
    return [s1 , s2]



async def main():
    #await asyncio.gather(*[test(f) for f in sorted(os.listdir("/Users/emadhekar/erase_me/images/"))])
    tasks = []
    for i in sorted(os.listdir("/Users/emadhekar/erase_me/images/")):
            task = asyncio.create_task(test(i))
            tasks.append(task)
    responses = await asyncio.gather(*tasks)
    print(responses)        

#asyncio.run(main())


loop = asyncio.get_event_loop()
try:
    loop.run_until_complete(main())
finally:
    loop.run_until_complete(loop.shutdown_asyncgens())
    loop.close()
