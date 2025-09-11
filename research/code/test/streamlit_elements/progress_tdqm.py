from time import sleep
from stqdm import stqdm
import streamlit as st

st.title('test')
for _ in stqdm(range(400), desc='main'):
    for _ in stqdm(range(60), desc='secondary'):
        sleep(.5)

st.success('complete')        