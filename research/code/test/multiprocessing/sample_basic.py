import asyncio
from aiomultiprocess import Pool
import logging

class MyObject:
    def __init__(self, name):
        self.name = name
        logging.info(f"Object {self.name} initialized in process")

    async def my_method(self, value):
        await asyncio.sleep(0.1)
        return f"Process {self.name}: {value * 2}"

async def worker_function(obj, data):
    return await obj.my_method(data)

async def setup_logging(level=logging.WARNING):
    logging.basicConfig(level=level)

async def main():
    logging.basicConfig(level=logging.INFO)

    async with Pool(initializer=setup_logging, initargs=(logging.DEBUG,)) as pool:
        obj1 = MyObject("One")
        obj2 = MyObject("Two")
        tasks = [
            pool.apply(worker_function, (obj1, 5)),
            pool.apply(worker_function, (obj2, 10)),
        ]
        results = await asyncio.gather(*tasks)
        logging.info(f"Results: {results}")

if __name__ == "__main__":
    asyncio.run(main())