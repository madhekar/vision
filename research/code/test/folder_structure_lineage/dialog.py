import streamlit as st

@st.dialog("Enter your location")
def add_location():
    name = st.text_input('Enter name or description of location')
    street = st.text_input("Enter street address of location")
    city = st.text_input("Enter city/ town/ village of location")
    state = st.text_input("Enter state/ province name of location")
    country = st.text_input("Enter country name of location")
    pincode = st.text_input("Enter pin/ zip code of location")

    if st.button("Submit"):
        st.session_state.location = {"name": name, "street": street, 'city': city, 'state': state, 'country': country, 'zip code': pincode}
        st.rerun()

if "location" not in st.session_state:
    st.write("Add your Location details")
    if st.button("Add Location"):
        add_location()

else:
    f"Entered Location for {st.session_state.location['name']} with street address {st.session_state.location['street']}"