import asyncio
import random

import aiomultiprocess


async def coro_func(value: int) -> int:
    await asyncio.sleep(random.randint(1, 3))
    return value * 2


async def main():
    results = []
    async with aiomultiprocess.Pool() as pool:
        async for result in pool.map(coro_func, [1, 2, 3]):
            results.append(result)

        # The result depends on the order in which the parameters are passed in,
        # not on which task end first
        # Output: [2, 4, 6]
        print(results)


if __name__ == "__main__":
    asyncio.run(main())