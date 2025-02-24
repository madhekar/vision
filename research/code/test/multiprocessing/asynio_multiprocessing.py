import asyncio
import multiprocessing
import time
import os

async def do_work(x):
    await asyncio.sleep(1)
    return x * x

async def run_in_process(func, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, func, *args)

def process_task(x):
    time.sleep(1)
    pid = os.getpid()
    print(f"Task {x} processed by PID {pid}")
    return x * x

async def main():
    with multiprocessing.Pool(processes=4) as pool:
        tasks = [run_in_process(process_task, i) for i in range(8)]
        results = await asyncio.gather(*tasks)
        print("Results:", results)

if __name__ == "__main__":
    asyncio.run(main())