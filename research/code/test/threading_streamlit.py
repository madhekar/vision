import threading
import streamlit as st
import time
import queue
import sys


q = queue.Queue()


def test_run():
    for x in range(1, 10):
        val = x
        multiply = val * 10
        q.put((val, multiply))
        print(val, ":", multiply)
        time.sleep(1)
        if x == 8:
            #threading.current_thread().terminate()
            sys.exit()


def update_dashboard():
    while True:
        val, multiply = q.get()

        col1, col2 = st.columns(2)
        col1.metric(label="Current Value", value=val)
        col2.metric(label="Multiply by 10 ", value=multiply)


threading.Thread(target=test_run, daemon=True).start()

# dashboard title
st.title("Streamlit Learning")

with st.empty():
    update_dashboard()
     
    st.write("started thread!cd ..") 
    