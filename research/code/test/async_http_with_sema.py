import asyncio
from aiohttp import ClientSession


async def fetch(url, session):
    async with session.get(url) as response:
        data = await response.read()
        print(data)
        return data


async def bound_fetch(sem, url, session):
    async with sem:
        return await fetch(url, session)


async def run(r):
    url = "http://localhost:8080"
    tasks = []
    sem = asyncio.Semaphore(1000)

    async with ClientSession() as session:
        for i in range(r):
            task = asyncio.ensure_future(bound_fetch(sem, url, session))
            tasks.append(task)

        responses = asyncio.gather(*tasks)
        await responses
        print(responses.result())

number = 10
loop = asyncio.get_event_loop()

future = asyncio.ensure_future(run(number))
loop.run_until_complete(future)