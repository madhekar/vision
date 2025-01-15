import os
import pandas as pd
import streamlit as st


def update(sap, saf, edit_df):
   print(sap + ':' + saf)
   edit_df.to_csv(os.path.join(sap, saf), index=False)


def read_addresses_file(sap, saf):
    df = pd.read_csv(os.path.join(sap, saf),delimiter=",")

    if "df_val" not in st.session_state:
        st.session_state.df_val = df
    return df

def show_current_addresses(df):
    show_df = st.table(df)
    return show_df

def show_edit_addresses(sap, saf, df):
    edit_df = st.data_editor(df, key="editor", num_rows="dynamic")
    if edit_df is not None and not edit_df.equals(st.session_state["df_val"]):
        update(sap, saf, edit_df)
        st.session_state["df_val"] = edit_df
    return edit_df

if __name__=='__main__':
    zapp_static_locaton_prep_data_path = "/home/madhekar/work/home-media-app/data/input-data/prep/"
    zapp_static_locaton_prep_file = "static_addresses.csv"

    st.set_page_config(layout="wide")

    df = read_addresses_file(zapp_static_locaton_prep_data_path, zapp_static_locaton_prep_file)
    c1, c2 =st.columns([1,2], gap="small")
    with c1:
        show_edit_addresses(zapp_static_locaton_prep_data_path, zapp_static_locaton_prep_file, df)

    with c2:
       show_current_addresses(df)