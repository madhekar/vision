import streamlit as st
import pandas as pd

mycsv = 'my.csv'

def update(editdf):
    editdf.to_csv(mycsv, index=False)
    
@st.cache_data(ttl='1d')
def load_df():
    return pd.read_csv(mycsv)

df = load_df()
edf = st.data_editor(df)  

st.button('save', on_click=update, args=(edf, ))
