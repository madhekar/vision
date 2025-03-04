import asyncio
import aiomultiprocess as amp

async def task_one(x):
    return x * 2


async def task_two(x, y):
    return x + y


async def main():
    list1 = [10, 20, 30, 200, 100, 400]
    list2 = [40, 50, 60, 400, 500, 800]
    
    combined_list = list(zip(list1, list2))
    print(combined_list)
    async with amp.Pool(16) as pool:
        results = await asyncio.gather(
            pool.map(task_one, [1, 2, 3]),
            pool.starmap(task_two, [(20,30), (45,55),(78,90)])
        )
    print(results)

if __name__ == "__main__":
    asyncio.run(main())