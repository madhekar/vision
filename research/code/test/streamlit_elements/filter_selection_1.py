import altair as alt
from vega_datasets import data
import streamlit as st

# Load the data
stocks = data.stocks.url

# Define an interval selection (brush and drag)
# This selection will be applied across all linked charts by default
# "symbol" makes the selection specific to each line/symbol
selector = alt.selection_single(fields=['symbol'])

# Chart 1: The main interactive chart
chart = alt.Chart(stocks).mark_line().encode(
    x='date:T',
    y='price:Q',
    color='symbol:N',
    # Use the selection to change opacity based on whether the item is selected
    opacity=alt.condition(selector, alt.value(1), alt.value(0.5))
).add_selection(
    selector # Add the selection to the chart so users can interact with it
)

# Chart 2: A filtered view of the data
# This chart uses transform_filter to show ONLY the data corresponding to the selection
filtered_chart = alt.Chart(stocks).mark_point().encode(
    x='date:T',
    y='price:Q',
    color='symbol:N'
).transform_filter(
    # The key is to pass the selection object here
    selector
)
ch = (chart | filtered_chart).properties(
    title='Interactive Filter Example'
)
# Combine the charts
st.altair_chart(ch)

