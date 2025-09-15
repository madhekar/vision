import streamlit as st
import multiprocessing as mp
import time

def worker_function(queue, total_tasks):
    for i in range(total_tasks):
        time.sleep(0.1)  # Simulate work
        queue.put(i + 1) # Send progress update
    queue.put("DONE") # Signal completion

if __name__ == "__main__":
    st.title("Multiprocessing Progress Bar Example")

    total_tasks = 100
    progress_queue = mp.Queue()

    if st.button("Start Process"):
        # Start the worker process
        process = mp.Process(target=worker_function, args=(progress_queue, total_tasks))
        process.start()

        progress_bar = st.progress(0, text="Starting...")

        current_progress = 0
        while current_progress < total_tasks:
            update = progress_queue.get()
            if update == "DONE":
                break
            current_progress = update
            progress_bar.progress(current_progress / total_tasks, text=f"Processing: {current_progress}/{total_tasks}")
            time.sleep(0.01) # Small delay to allow UI update

        progress_bar.progress(1.0, text="Process Complete!")
        process.join() # Wait for the worker process to finish
        st.success("Multiprocessing task finished!")