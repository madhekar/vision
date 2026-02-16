import streamlit as st

# Customizing the 3-dot (hamburger) menu
st.set_page_config(
    page_title="Custom Menu",
    menu_items={
        'Get Help': 'https://www.google.com',
        'Report a bug': None,  # Hides the "Report a bug" option
        'About': "# This is a custom header! \nThis app has a custom menu."
    }
)

st.title("Three Dots Menu Customization")
st.write("Check the top right hamburger menu.")