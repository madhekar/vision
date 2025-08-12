import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")
mycsv = "my.csv"


def update(editdf):
    print(editdf)
    editdf.dropna(how="all", inplace=True)
    editdf.to_csv(mycsv, index=False)


def load_df():
    return pd.read_csv(mycsv)


df = load_df()
st.subheader("Location Editor")
edf = st.data_editor(
    df,
    column_config={
        "name": st.column_config.TextColumn(
            "Name",
            help="name or description of this location",
            max_chars=100,
            # validate="/^[A-Za-z0-9_@./#&+-]*$/",
        ),
        "address": st.column_config.TextColumn(
            "Address",
            help="physical address of this location",
            max_chars=100,
            # validate="/^\s*\S+(?:\s+\S+){2}/",
        ),
    },
    use_container_width=True,
    num_rows="dynamic",
    width=800,
)

st.button("save", on_click=update, args=(edf,))
