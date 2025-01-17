import pandas as pd
import numpy as np
import streamlit as st

if "df_value" not in st.session_state:
    st.session_state.df_value = None


def load_csv():
    df = pd.read_csv('my.csv', delimiter=',')
    st.session_state.df_value = df


def update(d_df):
    # d_df.to_csv('my.csv', index=False)
    # some more code
    clean_dataframe(d_df)
    #print(d_df)

def clean_dataframe(d):
    df_r = d
    df_r['name'].replace('',np.nan, inplace=True)
    df_r["address"].replace("", np.nan, inplace=True)
    df_r.dropna(subset=['name'], inplace=True)
    df_r.dropna(subset=['address'], inplace=True)

    print(df_r)

load_csv()

edited_df = st.data_editor(
    st.session_state.df_value,
    key="editor",
    num_rows="dynamic",
    #on_change=update,
    #args=(df,)
)

if edited_df is not None and not edited_df.equals(st.session_state["df_value"]):
    st.write('-----> here')
    # This will only run if
    # 1. Some widget has been changed (including the dataframe editor), triggering a
    # script rerun, and
    # 2. The new dataframe value is different from the old value
    update(edited_df)
    st.session_state["df_value"] = edited_df

