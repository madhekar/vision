import asyncio
import streamlit as st

def init_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop


async def get_data():
    # Emulating a fetch from a remote server
    data = [{"name": "Jane", "age": 19}, {"name": "John", "age": 24}]
    await asyncio.sleep(30)
    st.write(data)
    print(data)
    #return data


def async_main():
    loop = init_loop()
    task = loop.create_task(get_data())
    loop.run_until_complete(task)
    loop.close()


btn = st.button(label="call async")
if btn:
      async_main()
      st.write("done")  