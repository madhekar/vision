import streamlit as st
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import time
import pandas as pd
import pyap as ap
from pprint import pprint

if "parse" not in st.session_state:
   st.session_state['parse'] = False

if "latlon" not in st.session_state:
  st.session_state['latlon'] = False

if 'submit' not in st.session_state:
    st.session_state['submit'] = False 

if 'p_location' not in st.session_state:
    st.session_state['p_location'] = {}

# if 'location' not in st.session_state:
#      st.session_state.location = {}    


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

    # dfr = pd.DataFrame(
    #     locs, columns=["id", "country", "state", "desc", "lat", "lon"]
    # ).set_index("id", drop=True)
    # pprint(dfr.head())

    return d

def parse_address(full_add):
    par_address = ap.parse(full_add, country="US")
    return par_address[0].as_dict()
    # for add in par_address:
    #     print(add.as_dict())

@st.dialog("Enter your location")
def add_location():
    c1,c2 = st.columns([.25, 1])
    with c1:
       st.subheader("name")
    with c2:   
       name = st.text_input('name', placeholder='Enter name or description of location', label_visibility="collapsed")

    c11, c21 = st.columns([0.25, 1])
    with c11:
       st.subheader("address")
    with c21:   
       full_add = st.text_input('full address', placeholder='Enter Address of location', label_visibility="collapsed").strip()

    if st.button('Parse'):
        apa = parse_address(full_add)            
        st.subheader("location")
        if apa:  
            c1, c2, c3 , c4, c5 = st.columns([9,2,1,1,2], gap="small")
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
            st.session_state['parse'] = not st.session_state['parse']
   
    if st.session_state['parse']:
        if st.button('GetLatLon'): 
            print("+++", st.session_state["p_location"])
            fs = st.session_state["p_location"]
            st.text_input(label="query", value=fs, placeholder=fs)
            ld= location_details(name=name, query=fs)
            st.text_input(label='location details', value=str(ld), placeholder=str(ld))
            st.session_state['location'] = ld
            st.session_state['latlon'] = not st.session_state['latlon']

    if st.session_state['parse']:
        if st.session_state['latlon']:
            if st.button("Submit"):
                st.session_state["submit"] = not st.session_state["submit"]
                #st.rerun()
                st.session_state["parse"] = False
                st.session_state["latlon"] = False
                st.session_state["submit"] = False

if "location" not in st.session_state:
    st.write("Add your Location details")
    if st.button("Add Location"):
        add_location()

else:
    f"Entered Location for {st.session_state.location} with street address {st.session_state.location}"