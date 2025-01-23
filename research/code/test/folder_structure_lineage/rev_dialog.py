import streamlit as st
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import time
import pandas as pd
import pyap as ap
from pprint import pprint

if 'stage' not in st.session_state:
    st.session_state.stage = 0

def set_state(i):
    st.session_state.stage = i

def parse_address(full_add):
    par_address = ap.parse(full_add, country="US")
    return par_address[0].as_dict()

def state_changes():

    if st.session_state.stage == 0:
        name = st.text_input('Name')
        address = st.text_input('Address', on_change=set_state, args=[1])

    if st.session_state.stage >= 1:
        st.write(f'Hello {name}!')
        color = st.selectbox(
            'Pick a Color',
            [None, 'red', 'orange', 'green', 'blue', 'violet'],
            on_change=set_state, args=[3]
        )
        if color is None:
            set_state(2)

    if st.session_state.stage >= 2:
        st.write(f':{color}[Thank you!]')
        st.button('Start Over', on_click=set_state, args=[0])
        

state_changes()        