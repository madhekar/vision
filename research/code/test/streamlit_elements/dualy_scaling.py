import altair as alt
from vega_datasets import data
import pandas as pd
import streamlit as st


base = alt.Chart(data.cars()).mark_line().transform_fold(
    ['Horsepower', 'Miles_per_Gallon'],
    as_=['Measure', 'Value']
).encode(
    alt.Color('Measure:N'),
    alt.X('year(Year):T')
)

line_A = base.transform_filter(
    alt.datum.Measure == 'Horsepower'
).encode(
    alt.Y('average(Value):Q').title('Horsepower')
)

line_B = base.transform_filter(
    alt.datum.Measure == 'Miles_per_Gallon'
).encode(
    alt.Y('average(Value):Q').title('Miles_per_Gallon')
)

ch = alt.layer(line_A, line_B).resolve_scale(y='independent')

st.altair_chart(ch)