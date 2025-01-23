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

geolocator = Nominatim(user_agent="name_ofgent")


def location_details(name, query={}):
    locs = []
    d = {}
    try:
        loc = geolocator.geocode(
            query=query, exactly_one=True, timeout=60, addressdetails=True
        )
        data = loc.raw
        data = data["address"]
        state = country = ""
        state = str(data["state"]).lower()
        country = str(data["country_code"]).lower()
        d["id"] = 1
        d["desc"] = name
        d["state"] = state
        d["country"] = country
        d["lat"] = loc.latitude
        d["lon"] = loc.longitude
        locs.append(d)
    except Exception as e:
        pprint(f"error: geocode failed with {query} with exception {e}")
    return d

def parse_address(full_add):
    par_address = ap.parse(full_add, country="US")
    print(par_address)
    return par_address[0].as_dict()

def set_state(i):
    st.session_state.stage = i

if st.session_state.stage == 0:
    st.button('Begin', on_click=set_state, args=[1])

if st.session_state.stage >= 1:
    name = st.text_input('Name')
    address = st.text_input('Address', on_change=set_state, args=[2])

if st.session_state.stage >= 2:
    st.write(f'Hello {name} {address}!')
    pa = parse_address(address)
    st.write(pa)
    bchecked = st.checkbox(label='is correct?')
    if bchecked:
        set_state(3)
    # color = st.selectbox(
    #     'Pick a Color',
    #     [None, 'red', 'orange', 'green', 'blue', 'violet'],
    #     on_change=set_state, args=[3]
    # )
    if pa is None:
        set_state(2)

if st.session_state.stage >= 3:
    ld = location_details(name=name, query=pa)
    st.write(ld)
    st.write(f'Thank you!')
    st.button('Start Over', on_click=set_state, args=[0])
