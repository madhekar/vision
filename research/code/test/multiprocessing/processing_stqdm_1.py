from multiprocessing import Pool, freeze_support
from time import sleep

import streamlit as st

# https://discuss.streamlit.io/t/stqdm-a-tqdm-like-progress-bar-for-streamlit/10097
# pip install stqdm
from stqdm import stqdm

c1, c2, _ = st.columns([1,1,1])

num_p = c1.number_input('Number of processes', value=5, min_value=1, max_value=100)
num_i = c2.number_input('Number of iterations', value=100, min_value=1, max_value=1000, step=100)

message = st.empty()
stqdm_container = st.container()

def sleep_and_return(i):
    sleep(0.25)
    return i


def run_pool(n_processes=2, n_iterations=10):
    with stqdm_container:
        with Pool(processes=n_processes) as pool:
            for i in stqdm(pool.imap(sleep_and_return, range(n_iterations)), total=n_iterations):
                message.success(f'Iteration: {i}')

if __name__ == '__main__':
    # On Windows an error was being thrown which suggested freeze_support() as a fix
    #   "An attempt has been made to start a new process before the
    #    current process has finished its bootstrapping phase...."
    freeze_support()

    option = st.radio('Select an option', ['Run forever', 'Run once', 'Stop'], index=2, horizontal=True, key='options')

    # Run forever (for demo purposes)
    if option == 'Run forever':
        while True:
            run_pool(n_processes=num_p, n_iterations=num_i)
    # More realistic use case to run once and stop
    elif option == 'Run once':
        run_pool(n_processes=num_p, n_iterations=num_i)

    message.info('Waiting to run again...')