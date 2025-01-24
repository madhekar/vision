import streamlit as st
from geopy.geocoders import Nominatim
import pyap as ap
from pprint import pprint

geolocator = Nominatim(user_agent="name_ofgent")

if 'stage' not in st.session_state:
    st.session_state.stage = 0

def set_state(i):
    print('state: ', str(i))
    st.session_state.stage = i

def parse_address(full_add):
    par_address = ap.parse(full_add, country="US")
    print(par_address)
    return par_address[0].as_dict()

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
        st.error(f"error: geocode failed with {query} with exception {e}", icon=":material/error:")
    return d


def display_parsed_address(apa): 
    d={} 
    d['street'] =  apa['full_street']
    d['city'] = apa['city']       
    d['state'] = apa['region1']
    d['country'] = apa['country_id']
    d['postalcode'] = apa['postal_code']
    st.info(f"{d['street']} {d['city']} {d['state']} {d['country']} {d['postalcode']}", icon=":material/check:")
    return d

@st.dialog("Enter location details: ")
def add_location():

    if st.session_state.stage >= 0:
        name = st.text_input('Name')
        address = st.text_input('Address', on_change=set_state, args=[1])

    if st.session_state.stage >= 1:
        try:
            pa = parse_address(address)
            if pa is None:
                set_state(1)
            else:
                d = display_parsed_address(pa)
                st.info(f'Search Latitude and Longitude for : {name}', icon=":material/search:")
                set_state(2)
        except Exception as e:
                st.error(f'Failed {e} to validate address for location: {name}', icon=":material/error:")  

    if st.session_state.stage >= 2:
        try:
            ld= location_details(name=name, query=d)
            st.info(f'Latitude: {str(ld["lat"])} and Longitude: {str(ld["lon"])} found for : {name}', icon=":material/done_all:")
        except Exception as e:
            st.error(f'Failed {e} to search latitude and longitude for location: {name}', icon=":material/error:") 

        c1, c2 = st.columns([1,1],gap="small")
        with c1:
            st.button('submit')
        with c2:
            st.button('Cancel', on_click=set_state, args=[0])

add_location()              