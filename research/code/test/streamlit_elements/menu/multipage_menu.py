import streamlit as st
from streamlit_option_menu import option_menu

# Import your page files as modules
import home_page
import data_analysis
import settings

st.set_page_config(layout="wide") # Optional: for wider layout

with st.sidebar:
    selected = option_menu(
        menu_title="ZMedia",  # Required
        options=["Home", "Data Analysis", "Settings"],  # Required
        icons=["house", "bar-chart", "gear"],  # Optional: Bootstrap icons
        menu_icon="cast",  # Optional: Icon for the menu title
        default_index=0,  # Optional: Default selected option
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "red", "font-size": "20px"},
            "nav-link": {
                "font-size": "20px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#eee",
            },
            "nav-link-selected": {"background-color": "A5BFA6"},
        },
    )

# Logic to display the selected page
if selected == "Home":
    home_page.app() # Call the function that runs the home page content
elif selected == "Data Analysis":
    data_analysis.app()
elif selected == "Settings":
    settings.app()

