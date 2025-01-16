import os
import pandas as pd
import streamlit as st

sap = "/home/madhekar/work/home-media-app/data/input-data/prep/"
saf = "static_addresses.csv"

def update(edit_df):
   print(sap + ':' + saf)
   edit_df.to_csv(os.path.join(sap, saf), index=False)


def read_addresses_file():
    df = pd.read_csv(os.path.join(sap, saf),delimiter=",")

    if "df_val" not in st.session_state:
        st.session_state.df_val = df
    return df

def show_current_addresses(df):
    show_df = st.table(df)
    return show_df

def show_edit_addresses(df):
    edit_df = st.data_editor(df, key="editor", num_rows="dynamic")
    if edit_df is not None and not edit_df.equals(st.session_state["df_val"]):
        update(edit_df)
        st.session_state["df_val"] = edit_df
        return edit_df

if __name__=='__main__':

    st.set_page_config(layout="wide")

    df = read_addresses_file()
    c1, c2 =st.columns([3,1], gap="small")
    with c1:
        show_edit_addresses(df)

    with c2:
       show_current_addresses(df)