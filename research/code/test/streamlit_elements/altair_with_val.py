import altair as alt
from vega_datasets import data
import pandas as pd
import streamlit as st


source = data.wheat()
images = [{"source": "Berkeley", "count": 580},{"source": "madhekar", "count": 1020},{"source": "samsung", "count": 200}]
df = pd.DataFrame(images)
print(df)
base = alt.Chart(df).encode(
    x='source:N',
    y="count:Q",
    text='count',
    color='source:N'
)
ch = base.mark_bar() + base.mark_text(align='center', dy=-10)



st.altair_chart(ch, use_container_width=True)

data = alt.graticule(step=[15,15])
gc = alt.Chart(data).mark_geoshape(stroke="blue").project('orthographic', rotate=(0, -45,0))

st.altair_chart(gc)

