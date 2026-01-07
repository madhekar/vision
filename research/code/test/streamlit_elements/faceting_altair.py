import altair as alt
from vega_datasets import data
import streamlit as st

source = data.population.url

print(data.population().head(10))
base = alt.Chart(source).mark_area().encode(
    x="age:O",
    y="people:Q",
).properties(
    width=180,
    height=180
)

# Apply the filter to the base chart before faceting
filtered_chart = base.transform_filter(
    alt.datum.year <= 2000 # Filters for data where the 'year' field is 2000
)

# Facet the filtered chart by sex
final_chart = filtered_chart.facet(
    column='sex:N'
)

st.altair_chart(final_chart)
