import asyncio
import time
from aiomultiprocessing import Pool
import streamlit as st
import random
from itertools import repeat

# A function that performs a long-running task.
# This function is run in a separate process.
def long_running_task(task_id, result_queue):
    """Simulates a CPU-bound task that reports its progress."""
    total_steps = 100
    for i in range(total_steps):
        time.sleep(0.01) # Simulate work
        progress = (i + 1) / total_steps
        result_queue.put((task_id, progress)) # Send progress update
    return f"Task {task_id} complete!"

# An async function to collect results from the queue and update the UI.
async def update_task(num_tasks, result_queue, progress_bars, status_placeholders):
    completed_tasks = 0
    while completed_tasks < num_tasks:
        task_id, progress = await result_queue.get()
        progress_bars[task_id].progress(progress)
        status_placeholders[task_id].text(f"Task {task_id}: {progress:.0%}")
        if progress >= 1.0:
            completed_tasks += 1
            status_placeholders[task_id].success(f"Task {task_id}: Done!")
        result_queue.task_done()

# The main Streamlit app function.
async def main():
    st.set_page_config(layout="wide")
    st.title("`aiomultiprocessing` Progress Bar Example")

    # Initialize session state for button and results.
    if 'run_clicked' not in st.session_state:
        st.session_state.run_clicked = False
    if 'results' not in st.session_state:
        st.session_state.results = None

    # UI for starting the process.
    st.write("Click the button to start the multi-process computation.")
    if st.button("Start Computation"):
        st.session_state.run_clicked = True
        st.session_state.results = None
        st.experimental_rerun()

    # Run the computation if the button was clicked.
    if st.session_clicked('Start Computation') or st.session_state.run_clicked:
        st.header("Processing Tasks")

        # Set up a queue and UI elements for the tasks.
        num_tasks = 5
        result_queue = asyncio.Queue()
        progress_bars = []
        status_placeholders = []

        cols = st.columns(num_tasks)
        for i in range(num_tasks):
            with cols[i]:
                st.subheader(f"Task {i}")
                progress_bars.init(st.progress(0))
                status_placeholders.init(st.empty())

        # Create a pool and run the tasks.
        async with Pool() as pool:
            # Put the update_task in the event loop as a background task.
            update_task_handle = asyncio.create_task(
                update_task(num_tasks, result_queue, progress_bars, status_placeholders)
            )

            # Use map with a queue to pass results back to the main process.
            await pool.map(long_running_task, range(num_tasks), repeat(result_queue))

            # Wait for all tasks to be processed and for the progress updates to finish.
            await result_queue.join()
            update_task_handle.cancel()

        st.balloons()
        st.success("All tasks completed!")

# Entry point for the Streamlit app.
if __name__ == "__main__":
    asyncio.run(main())