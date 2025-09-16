import time
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed

def fetch(i: int) -> int:
    time.sleep(0.1)
    return i

def _task(x):
    result = fetch(x)
    return x, result

counts = range(1000)

def calculate():
    with st.spinner("Running..."):
        with ThreadPoolExecutor() as executor:
            bar = st.progress(0)
            placeholder = st.empty()
            futures = {executor.submit(_task, count): count for count in counts}
            results = []
            for idx, future in enumerate(as_completed(futures), start=1):
                count, result = future.result()
                results.append((count, result))
                progress = idx / len(counts)
                placeholder.text(f"{int(progress * 100)}%")
                # update progress bar
                bar.progress(progress)

if st.button('Calculate!'):
    calculate()