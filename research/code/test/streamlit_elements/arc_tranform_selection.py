import altair as alt
from vega_datasets import data

import streamlit as st

# Load a sample dataset (e.g., population data)
source = data.population.url

# 1. Define a selection (e.g., single selection on 'year')
selection = alt.selection_point(fields=['year'], on='click', empty=False) # empty=False ensures something is selected by default

# 2. Create the base chart structure
base = alt.Chart(source).encode(
    theta=alt.Theta("sum(people):Q", stack=True), # Use sum(people) for arc size
    color=alt.Color("year:N") # Color by year
)

# 3. Create a chart that uses mark_arc and is filtered by the selection
# This chart shows a "details" view, maybe another breakdown
filtered_arcs = base.mark_arc(outerRadius=120).transform_filter(
    selection # Apply the selection filter
).properties(
    title='Filtered Arc Chart Details'
)

# 4. Create a control chart (e.g., a bar chart) to make the selection
# The selection logic is added here.
selection_control = alt.Chart(source).mark_bar().encode(
    x='year:O',
    y='sum(people):Q',
    color=alt.condition(
        selection, # If selected, color 'steelblue'
        alt.value('steelblue'),
        alt.value('lightgray') # Otherwise 'lightgray'
    )
).add_params(
    selection # Add the selection parameter to the control chart
).properties(
    title='Select Year to Filter Arcs'
)

# Combine and display the charts (e.g., horizontally concatenated)
ch = (selection_control | filtered_arcs).resolve_scale(color='independent')

st.altair_chart(ch)