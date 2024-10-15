import asyncio as aio
import streamlit as st
import LLM

def init_loop():
    loop = aio.new_event_loop()
    aio.set_event_loop(loop)
    return loop


async def get_data(
    sim,
    model,
    processor,
    top,
    temperature,
    question,
    article,
    location
):
    data = await LLM.fetch_llm_text(
        sim,
        model,
        processor,
        top,
        temperature,
        question, 
        article,
        location,
    )
    return data

def async_main(sim, model, processor, top, temperature, question, article, location):
    loop = init_loop()
    task = loop.create_task(
        get_data(sim, model, processor, top, temperature, question, article, location)
    )
    loop.run_until_complete(task)
    st.session_state["llm_text"] = task.result()
    loop.close()
