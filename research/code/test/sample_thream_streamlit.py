import streamlit as st
import threading
import time


class SomeThread(threading.Thread):
    def __init__(self, dummy_list):
        super(SomeThread, self).__init__(daemon=True)
        self.dummy_list = dummy_list
        self.sleep_time = 1
        self.shutdown = False

    def run(self):
        while not self.shutdown:
            print(self.dummy_list)
            time.sleep(2)

    def terminate(self):
        self.shutdown = True


@st.cache(allow_output_mutation=True)
def get_manager():
    return Manager()


# @st.cache(allow_output_mutation=True)
class Manager:
    def __init__(self):
        print("Manager is beeing Initialized...")
        self.dummy_list = []
        self.thread = SomeThread(self.dummy_list)
        self.thread.start()
        time.sleep(5)

    def do_smth(self, item):
        self.dummy_list.append(item)
        print("blub")

    def terminate(self):
        self.thread.terminate()


def main():
    # manager = Manager()
    manager = get_manager()
    selected = st.selectbox("Dummy", [1, 23, 4, 5, 8, 9, 34])
    if st.button("add"):
        manager.do_smth(selected)
        print('started')
        st.write('...started')


if __name__ == "__main__":
    main()
