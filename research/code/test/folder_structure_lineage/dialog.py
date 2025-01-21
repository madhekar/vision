import streamlit as st
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import time
import pandas as pd
from pprint import pprint

if "parse" not in st.session_state:
   st.session_state['parse'] = False

if "latlon" not in st.session_state:
  st.session_state['latlon'] = False

if 'submit' not in st.session_state:
    st.session_state['submit'] = False 

# if 'location' not in st.session_state:
#     st.session_state.location = {}    

d = []
geolocator = Nominatim(user_agent="name_ofgent")

def location_details(name, query = {}):
    locs = []
    d = {}
    try:
        loc = geolocator.geocode(query=query, exactly_one=True, timeout=60, addressdetails=True)
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
        pprint(
            f"error: geocode failed with {query} with exception {e}"
        )

    dfr = pd.DataFrame(
        locs, columns=["id", "country", "state", "desc", "lat", "lon"]
    ).set_index("id", drop=True)
    pprint(dfr.head())

    return dfr

def parse_address(full_add):
    tokens = full_add.split(',')
    return tokens

@st.dialog("Enter your location")
def add_location():
    st.subheader("location name/ desc")
    name = st.text_input('name', placeholder='Enter name or description of location', label_visibility="collapsed")


    st.subheader("location address")
    full_address = st.text_input('full_address', placeholder='Enter Address of location', label_visibility="collapsed")

    if st.button('parse'):
        s, c, st, cn, po = parse_address(full_address)
        d={}
        street = st.text_input("Enter street address of location")
        d['street'] = street
        city = st.text_input("Enter city/ town/ village of location")
        d['city'] = city
        state = st.text_input("Enter state/ province name of location")
        d['state'] = state
        country = st.text_input("Enter country name of location")
        d['country'] = country
        pincode = st.text_input("Enter pin/ zip code of location")
        d['postalcode'] = pincode
    

    if st.button('latlon'):
        fs = d
        st.text_area(label="query", value=fs)
        ld= location_details(name=name, query=d)
        st.text_area(label='location details', value=ld.to_string())
        st.session_state['latlon'] = not st.session_state['latlon']

    if st.session_state['latlon']:
        if st.button("Submit"):
            st.session_state.location = {"nam": name, "street": street, 'city': city, 'state': state, 'country': country, 'zip code': pincode}
            st.session_state["submit"] = not st.session_state["submit"]
            #st.rerun()


if "location" not in st.session_state:
    st.write("Add your Location details")
    if st.button("Add Location"):
        add_location()

else:
    f"Entered Location for {st.session_state.location['nam']} with street address {st.session_state.location['street']}"