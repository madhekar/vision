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
    print(par_address)
    return par_address[0].as_dict()

def display_parsed_address(apa):
    c1, c2, c3 , c4, c5 = st.columns([.7,.5,.3,.3,.5], gap="small")
    d={}  
    with c1:
        d['street'] =  apa['full_street'] 
        street = st.text_input("street name", value=d['street'], placeholder=d['street'],label_visibility="collapsed")

    with c2:
        d['city'] = apa['city']
        city = st.text_input("city/ town/ village", value=d['city'], placeholder=d['city'],label_visibility="collapsed")

    with c3:    
        d['state'] = apa['region1']
        state = st.text_input("state/ province", value=d['state'], placeholder=d['state'],label_visibility="collapsed")
    
    with c4:
        d['country'] = apa['country_id']
        country = st.text_input("country code/ name", value=d['country'], placeholder=d['country'], label_visibility="collapsed")
    with c5:    
        d['postalcode'] = apa['postal_code']
        pincode = st.text_input("pin/ zip code", value=d['postalcode'], placeholder=d['postalcode'], label_visibility="collapsed")
    
    st.session_state['p_location'] = d



if st.session_state.stage == 0:
    name = st.text_input('Name')
    address = st.text_input('Address', on_change=set_state, args=[1])

if st.session_state.stage >= 1:
    st.write(f' hello{name}')
    # pa = parse_address(address)
    # d = display_parsed_address(pa)

    # st.write(f'Hello {name}!')
    # color = st.selectbox(
    #     'Pick a Color',
    #     [None, 'red', 'orange', 'green', 'blue', 'violet'],
    #     on_change=set_state, args=[3]
    # )
    # if color is None:
    #     set_state(2)

if st.session_state.stage >= 2:
    st.write(f'Thank you!')
    st.button('Start Over', on_click=set_state, args=[0])
        
      