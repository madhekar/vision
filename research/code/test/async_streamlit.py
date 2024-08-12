import streamlit as st
import asyncio
import random as r

val_one = 0
val_two = 0


st.title("Zesha Async")


async def zfWorker():
    global val_one
    while True:
        await asyncio.sleep(r.randint(1, 3))
        val_one = r.randint(1, 10)
        st.metric("First Worker Executed: ", val_one)


async def zsWorker():
    global val_two
    while True:
        await asyncio.sleep(r.randint(1, 3))
        val_two = r.randint(1, 10)
        st.metric("Second Worker Executed: ", val_two)


async def main():
    await asyncio.gather(zfWorker(), zsWorker())


if __name__ == "__main__":
    with st.empty():  # Modified to use empty container
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            pass
        finally:
            print("Closing Loop")
            loop.close()
