import altair as alt
from vega_datasets import data
import streamlit as st

cars = data.cars()

ch = alt.Chart(cars).mark_bar().encode(
    alt.X('Horsepower', axis=alt.Axis(title="HORSEPOWER")),
    alt.Y('Miles_per_Gallon', axis=alt.Axis(title="Miles Per Gallon")),
    color='Origin',
    shape='Origin'
).configure_axis(
    labelFontSize=20,
    titleFontSize=20
)

st.altair_chart(ch)