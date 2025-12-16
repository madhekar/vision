import altair as alt
from altair.datasets import data
import streamlit as st

penguins = data.penguins.url

chart1 = alt.Chart(penguins).mark_point().encode(
    x=alt.X('Flipper Length (mm):Q', scale=alt.Scale(zero=False)),
    y=alt.Y('Body Mass (g):Q', scale=alt.Scale(zero=False)),
    color='Species:N'
).properties(
    height=300,
    width=300
)

chart2 = alt.Chart(penguins).mark_bar().encode(
    x='count()',
    y=alt.Y('Body Mass (g):Q', bin=alt.Bin(maxbins=30)),
    color='Species:N'
).properties(
    height=300,
    width=100
)

st.altair_chart(chart1 | chart2)