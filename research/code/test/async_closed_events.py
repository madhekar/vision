import asyncio
import streamlit as st

async def f():
    st.write("starting...")
    await asyncio.sleep(10)
    st.write("ending...")

loop = asyncio.new_event_loop()
loop.run_until_complete(f())
loop.close()

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
