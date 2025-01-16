
import os
import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder


sap = "."
saf = "my.csv"

@st.cache_data
def read_addresses_file():
    df = pd.read_csv(os.path.join(sap, saf), delimiter=",")
    return df

df = read_addresses_file()

# Create grid options
gb = GridOptionsBuilder.from_dataframe(df)
gb.configure_pagination()
gb.configure_side_bar()
gridOptions = gb.build()

# Display the grid
g_response = AgGrid(df, gridOptions=gridOptions, update_mode='SELECTION_CHANGED', allow_unsafe_jscode=True)