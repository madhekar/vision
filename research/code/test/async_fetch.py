import asyncio

async def fetch_url_data(url):
    return f"Data from {url}"

async def async_main():
    urls = ["amazon.com", "google.com", "apple.com"]
    tasks = [fetch_url_data(u) for u in urls]

    for completed_task in asyncio.as_completed(tasks):
        result = await completed_task
        await asyncio.sleep(1)
        print(result)

asyncio.run(async_main())

print('do normal stuff..')