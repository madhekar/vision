import streamlit as st
import datetime

MIN_DT = datetime.datetime(1998, 1, 1)
MAX_DT = datetime.datetime.now()

def initUI():

    st.set_page_config(
        page_title="zesha: Home Media Portal (HMP)",
        page_icon="/home/madhekar/work/zsource/zesha-high-resolution-logo.jpeg",
        initial_sidebar_state="auto",
        layout="wide",
    )  # (margins_css)


    st.markdown(
        """
        <style>
            
        .reportview-container {
          margin-top: -2em;
          margin-right: -10em;
        }
            
        .block-container{
        padding-top: 2rem;
        padding-bottom:0rem;
        padding-left: 1rem;
        padding-right: 1rem;
        }
            
        MainMenu { visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        stDecoration {display: none;}
            [alt=Logo] {
            height: 6rem;
        }
            
        section[data-testid="stSidebar"] {
            width: 25rem !important;  # Set the width to your desired value 
        }

        .big-font {
           font-size:1.3rem;
        }

        .big-font-subh {
           font-size:1.3rem;
           color:blue;
           #font-weight: bold;
        }
                
        .big-font-header {
           font-size:1.5rem;
           color:blue;
           font-weight: bold;
        }        

        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size:1.3rem;
        color:blue;
        #font-weight:bold;    
        }
            
        .stTextInput > label {
        font-size:1.2rem;
        #font-weight:bold;
        color:blue;
        }

        .stMultiSelect > label {
        font-size:1.2rem;
        #font-weight:bold;
        color:blue;
        }

        .stSelectbox > label {
        font-size:1.2rem;
        #font-weight:bold;
        color:blue;
        }

        .stFileUploader > label {
        font-size:1.2rem;
        #font-weight:bold;
        color:blue;
        }

        .stSlider > label {
        font-size:1.2rem;
        #font-weight:bold;
        color:blue;
        }

        .stButton > label {
        font-size:1.2rem;
        #font-weight:bold;
        color:blue;
        }
        
        .stButton > button:first-child {
            height:3rem;
            width:5rem;
            font-size:1.3rem;
            #font-weight:bold;
            color:blue;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
