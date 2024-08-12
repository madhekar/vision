import streamlit as st
import asyncio
import random as r


st.title("Zesha Async (no global)")


async def zfWorker():
    await asyncio.sleep(r.randint(1, 10))
    val_one = r.randint(1, 100)
    st.metric("First Worker Executed: ", val_one)


async def zsWorker():
    await asyncio.sleep(r.randint(1, 15))
    val_two = r.randint(1, 200)
    st.metric("Second Worker Executed: ", val_two)


async def main():
    with st.empty():
        while True:
            left_col, right_col = st.columns(2)
            with left_col:
                await asyncio.gather(zfWorker())
            with right_col:
                await asyncio.gather(zsWorker())


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
