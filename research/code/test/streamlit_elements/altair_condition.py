import altair as alt
import pandas as pd
import streamlit as st

df = pd.DataFrame({
    'x': ['A', 'B', 'C', 'D', 'E'],
    'y': [5, 3, 6, 7, 2],
})

c= alt.Chart(df).mark_bar().encode(
    x='x',
    y=alt.Y('y', axis=alt.Axis(
        labelColor=alt.condition('datum.value > 3 && datum.value < 7', alt.value('red'), alt.value('black'))
    ))
)

st.altair_chart(c)