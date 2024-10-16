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
            height: 7rem;
        }
            
        section[data-testid="stSidebar"] {
            width: 25rem !important;  # Set the width to your desired value 
        }

        .big-font {
           font-size:1.3rem;
           color:#3E4D34;
           #font-weight:bold;
        }

        .big-font-subh {
           font-size:1.3rem;
           color:#5E734E;
           font-weight: bold;
        }
                
        .big-font-header {
           font-size:1.5rem;
           color:#3E4D34;
           font-weight: bold;
        }        

        .big-font-title {
           font-size:2rem;
           color:#3E4D34;
           font-weight: bold;
           text-align: center;
        } 

        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size:1.3rem;
        color:#5E734E;
        font-weight:bold;    
        }
            
        .stTextInput > label {
        font-size:1.2rem;
        font-weight:bold;
        color:#5E734E;
        }

        .stMultiSelect > label {
        font-size:1.2rem;
        font-weight:bold;
        color:#5E734E;
        }

        .stSelectbox > label {
        font-size:1.2rem;
        font-weight:bold;
        color:#5E734E;
        }

        .stFileUploader > label {
        font-size:1.2rem;
        font-weight:bold;
        color:#5E734E;
        }

        .stSlider > label {
        font-size:1.2rem;
        font-weight:bold;
        color:#5E734E;
        }

        .stButton > label {
        font-size:1.2rem;
        font-weight:bold;
        color:#5E734E;
        }
        
        .stButton > button:first-child {
            height:3rem;
            width:6rem;
            font-size:1.4rem;
            font-weight:bold;
            color:#5E734E;
            }

        # div[data-testid="column"]:nth-of-type(1)
        # {
        #     border: 1px solid black;
        # }    
        # div[data-testid="column"]:nth-of-type(2)
        # {
        #     border: 1px solid blue;
        #     text-align:end;
        # }

        # div[data-testid="stHorizontalBlock"]{
        #   display: flex
        # }

        # div[data-testid="column"]{
        #   flex:1;
        #   padding: 1em;
        #   border: solid;
        #   border-radius: 1px;
        #   border-color: gray;
        # }

        
 
        </style>
        """,
        unsafe_allow_html=True,
    )
