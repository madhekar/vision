import time
import streamlit as st
from aiomultiprocess import Pool
from multiprocessing import Manager

# A function to run in a separate process
def my_worker_task(task_id, progress_queue):
    total_steps = 10
    for i in range(total_steps):
        time.sleep(0.5)  # Simulate a time-consuming step
        progress_queue.put(1)  # Put a single update on the queue
    return f"Task {task_id} completed."

# A function that monitors the progress queue and updates the Streamlit UI
async def manage_progress(total_tasks, progress_queue):
    completed_tasks = 0
    progress_bar = st.progress(0, text="Starting tasks...")
    
    while completed_tasks < total_tasks:
        if not progress_queue.empty():
            progress_queue.get()  # Get and discard the update message
            completed_tasks += 1
            percent_complete = (completed_tasks / total_tasks) * 100
            progress_bar.progress(int(percent_complete), text=f"Processing... {completed_tasks}/{total_tasks}")
        
        time.sleep(0.5) # Yield to allow the Streamlit UI to update

    progress_bar.progress(100, text="All tasks completed!")
    return "All tasks finished."

async def main():
    st.title("Streamlit and aiomultiprocess with progress bar")
    
    # Use a Manager to create a queue that can be shared between processes
    with Manager() as manager:
        progress_queue = manager.Queue()
        num_tasks = 5
        
        if st.button("Start Multiprocessing Tasks"):
            st.session_state.run_tasks = True
            
        if st.session_state.get('run_tasks', False):
            # Start the background tasks
            async with Pool() as pool:
                tasks = [pool.apply(my_worker_task, args=(i, progress_queue)) for i in range(num_tasks)]
                
                # Start the progress bar and wait for it to complete
                await manage_progress(num_tasks, progress_queue)
                
                # Retrieve the results from the tasks
                results = await pool.join()
                for result in results:
                    st.write(result)
            
            st.session_state.run_tasks = False

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())