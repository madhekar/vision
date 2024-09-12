import asyncio


def parallel_map(func, iterable, concurrent_limit=2, raise_error=False):
    async def worker(input_queue: asyncio.Queue, output_queue: asyncio.Queue):

        while not input_queue.empty():
            idx, item = await input_queue.get()
            try:
                # Support both coroutine and function. Coroutine function I mean!
                if asyncio.iscoroutinefunction(func):
                    output = await func(item)
                else:
                    output = func(item)
                await output_queue.put((idx, output))

            except Exception as err:
                await output_queue.put((idx, err))

            finally:
                input_queue.task_done()

    async def group_results(input_size, output_queue: asyncio.Queue):
        output = {}  # using dict to remove the need to sort list

        for _ in range(input_size):
            idx, val = await output_queue.get()  # gets tuple(idx, result)
            output[idx] = val
            output_queue.task_done()

        return [output[i] for i in range(input_size)]

    async def procedure():
        # populating input queue
        input_queue: asyncio.Queue = asyncio.Queue()
        for idx, item in enumerate(iterable):
            input_queue.put_nowait((idx, item))
        
        # Remember size before using Queue
        input_size = input_queue.qsize()

        # Generate task pool, and start collecting data.
        output_queue: asyncio.Queue = asyncio.Queue()
        result_task = asyncio.create_task(group_results(input_size, output_queue))
        tasks = [
            asyncio.create_task(worker(input_queue, output_queue))
            for _ in range(concurrent_limit)
        ]
        
        # Wait for tasks complete
        await asyncio.gather(*tasks)
        
        # Wait for result fetching
        results = await result_task
        
        # Re-raise errors at once if raise_error
        if raise_error and (errors := [err for err in results if isinstance(err, Exception)]):
            # noinspection PyUnboundLocalVariable
            raise Exception(errors)  # It never runs before assignment, safe to ignore.

        return results

    return asyncio.run(procedure())

if __name__ == "__main__":
    import random
    import time

    data = [1, 2, 3]
    err_data = [1, 'yo', 3]

    def test_normal_function(data_, raise_=False):
        def my_function(x):
            t = random.uniform(1, 2)
            print(f"Sleep {t:.3} start")

            time.sleep(t)
            print(f"Awake after {t:.3}")

            return x * x

        print(f"Normal function: {parallel_map(my_function, data_, raise_error=raise_)}\n")

    def test_coroutine(data_, raise_=False):
        async def my_coro(x):
            t = random.uniform(1, 2)
            print(f"Coroutine sleep {t:.3} start")

            await asyncio.sleep(t)
            print(f"Coroutine awake after {t:.3}")

            return x * x

        print(f"Coroutine {parallel_map(my_coro, data_, raise_error=raise_)}\n")

    # Test starts
    print(f"Test for data {data}:")
    test_normal_function(data)
    test_coroutine(data)

    print(f"Test for data {err_data} without raise:")
    test_normal_function(err_data)
    test_coroutine(err_data)

    print(f"Test for data {err_data} with raise:")
    test_normal_function(err_data, True)
    test_coroutine(err_data, True)  # this line will not run, but works same.