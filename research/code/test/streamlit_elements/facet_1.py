import altair as alt
from vega_datasets import data
import streamlit as st

cars = data.cars()

# Create a selection (e.g., an interval selection for interactivity)
# The selection can then be used as a filter predicate
selection = alt.selection_interval(bind='scales')

base = alt.Chart(cars).mark_point().encode(
    x='Horsepower:Q',
    y='Miles_per_Gallon:Q',
    color='Origin:N',
    # Add the selection to the base chart for interactivity
    # A conditional opacity encoding is used to show the effect of the filter/selection
    opacity=alt.condition(selection, alt.value(1.0), alt.value(0.2))
).add_selection(
    selection
)

# Apply a global transform_filter and then facet
# For example, filtering out a specific origin, then faceting by another variable
chart = base.transform_filter(
    # Example: filter only cars with more than 100 horsepower
    alt.datum.Horsepower > 100
).facet(
    column='Origin:N' # Facet by the 'Origin' column
)

#chart.save('filtered_faceted_chart.json')
st.altair_chart(chart)