import asyncio
import random
import os

async def job(id):
    print('Starting job:',id )
    await asyncio.sleep(random.randint(1, 3))
    print('Finished job:', id)
    return id

async def main():
    # create a list of worker tasks
    coros = [job(f) for f in os.listdir('.')]
    
    # gather the results of all worker tasks
    results = await asyncio.gather(*coros)
    
    # print the results
    print(f'Results: {results}')

asyncio.run(main())
