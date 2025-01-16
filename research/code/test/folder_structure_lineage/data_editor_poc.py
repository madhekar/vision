import pandas as pd
import streamlit as st

df = pd.read_csv('my.csv', delimiter=',')

if "df_value" not in st.session_state:
    st.session_state.df_value = df


def update(edited_df):
    print(edited_df)
    edited_df.to_csv('my.csv', index=False)
    # some more code


edited_df = st.data_editor(
    df,
    key="editor",
    num_rows="dynamic",
)

if edited_df is not None and not edited_df.equals(st.session_state["df_value"]):
    # This will only run if
    # 1. Some widget has been changed (including the dataframe editor), triggering a
    # script rerun, and
    # 2. The new dataframe value is different from the old value
    update(edited_df)
    st.session_state["df_value"] = edited_df
