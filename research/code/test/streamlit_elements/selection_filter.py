import altair as alt
from vega_datasets import data
import streamlit as st

# Load the data
pop = data.population.url

# Create an interval selection
# This selection will be driven by interaction on the 'bottom' chart
selection = alt.selection_interval(encodings=['x'])

# Top chart (filtered by the selection)
top = alt.Chart(pop).mark_line().encode(
    x='age:O',
    y='sum(people):Q',
    color='year:O'
).properties(
    width=600,
    height=200,
    title='Population (filtered by year selection below)'
).transform_filter(
    # The selection object is used directly as the filter condition
    selection
)

# Bottom chart (drives the selection)
bottom = alt.Chart(pop).mark_bar().encode(
    x='year:O',
    y='sum(people):Q',
    # Color condition highlights selected years
    color=alt.condition(selection, alt.value('steelblue'), alt.value('lightgray'))
).properties(
    width=600,
    height=100,
    title='Select years to filter top chart'
).add_params(
    # Add the selection to this chart so it can be interacted with
    selection
)

# Combine the two charts vertically
st.altair_chart(top & bottom)
