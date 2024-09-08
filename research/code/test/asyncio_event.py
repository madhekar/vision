import asyncio


async def waiter(event, seq):
    print(f"A{seq}", end=" ")
    await event.wait()
    # surprise! event is not set in the statement following 'await event.wait()'
    assert not event.is_set()
    print(f"B{seq}", end=" ")


async def main():
    event = asyncio.Event()

    tset = set()
    for i in range(10):
        tset.add(asyncio.create_task(waiter(event, i)))

    print("go A:", end=" ")
    await asyncio.sleep(3)
    print()

    event.set()
    event.clear()

    print("go B:", end=" ")
    await asyncio.sleep(2)
    print()

    for t in tset:
        assert t.done()


asyncio.run(main())