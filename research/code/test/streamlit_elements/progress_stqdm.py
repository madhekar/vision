import streamlit as st
from stqdm import stqdm
import multiprocessing as mp
import time


def worker_function(item):
    """A function to be executed in a separate process."""
    time.sleep(0.1)  # Simulate some work
    return item * 2


if __name__ == "__main__":
    st.title("Multiprocessing with stqdm in Streamlit")

    items = list(range(100))
    num_processes = 4

    if st.button("Start Processing"):
        with mp.Pool(num_processes) as pool:
            # Use stqdm to wrap the results from the multiprocessing pool
            results = list(
                stqdm(
                    pool.imap(worker_function, items),
                    total=len(items),
                    desc="Processing items",
                )
            )
        st.write("Processing complete!")
        st.write(f"Results: {results[:10]}...")  # Display first 10 results
