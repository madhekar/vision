import asyncio
import streamlit as st

models = ["caption", "llm"]
async def call_model(num):
    st.write("Starting func {0}...".format(num))
    await asyncio.sleep(1)
    st.write("Ending func {0}...".format(num))

async def create_tasks_func():
    tasks = list()
    for i in range(len(models)):
        tasks.append(asyncio.create_task(call_model(models[i])))
    await asyncio.wait(tasks)

btn = st.button(label="call models")
if btn:
  loop = asyncio.new_event_loop()
  asyncio.set_event_loop(loop)
  loop.run_until_complete(create_tasks_func())
  st.rerun()
  #loop.close()