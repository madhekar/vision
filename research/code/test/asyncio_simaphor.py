import asyncio
import os

from aiodecorators import Semaphore


@Semaphore(10)
async def test(i):
    print(f"Start Task {i}")
    await asyncio.sleep(2)
    print(f"Finished Task {i}")


async def main():
    await asyncio.gather(*[test(f) for f in sorted(os.listdir('/Users/emadhekar'))])


asyncio.run(main())