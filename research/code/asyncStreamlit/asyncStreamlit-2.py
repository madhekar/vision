import streamlit as st
import asyncio
import random as r

st.title("Hello World")


async def firstWorker():
    await asyncio.sleep(r.randint(1, 3))
    val_one = r.randint(1, 10)
    st.metric("First Worker Executed: ", val_one)


async def secondWorker():
    await asyncio.sleep(r.randint(1, 3))
    val_two = r.randint(1, 10)
    st.metric("Second Worker Executed: ", val_two)


async def main():
    with st.empty():
        while True:
            left_col, right_col = st.columns(2)
            with left_col:
                await asyncio.gather(firstWorker())
            with right_col:
                await asyncio.gather(secondWorker())


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    finally:
        print("Closing Loop")
        loop.close()
