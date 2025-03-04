import asyncio
import time
from aiomultiprocess import Pool

async def square(x):
    await asyncio.sleep(1)
    return x * x

async def main():
    start_time = time.time()
    async with Pool(4) as pool:
        results = await pool.map(square, range(10))
        print(results)
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())