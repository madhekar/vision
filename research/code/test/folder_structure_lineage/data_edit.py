import os
import pandas as pd
import streamlit as st

sap = "."
saf = "my.csv"


def save_df(df):
    print(sap + ":" + saf)
    df.to_csv(os.path.join(sap, saf), index=False)

@st.fragment
def read_addresses_file():
    df = pd.read_csv(os.path.join(sap, saf), delimiter=",")
    return df


def show_current_addresses(df):
    st.table(df)

if __name__ == "__main__":
    st.set_page_config(layout="wide")

    df = read_addresses_file()

    show_current_addresses(df)

    with st.form ('address_edit', clear_on_submit=True):
        name_value = st.text_input(label='enter location name or description')
        addess_value = st.text_input(label='enter address of the location')
        submit = st.form_submit_button(label='submit')

    if submit:
        d = {}
        d['name'] = name_value
        d['address'] = addess_value
        df.loc[len(df)] = d
        print(df)
        # save_df(df)
        # st.rerun()