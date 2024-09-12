import asyncio

async def my_task(id, sem: asyncio.Semaphore):
  async with sem:  
    await asyncio.sleep(7)
    return f'I am number {id} + 5'


async def main():
    semaphore = asyncio.Semaphore(value=2)
    tasks = [my_task(id, semaphore) for id in range(100)]
    
    for coro in asyncio.as_completed(tasks):
        result = await coro
        print(result)

asyncio.run(main())