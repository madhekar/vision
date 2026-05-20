import streamlit as st
import altair as alt
import pandas as pd

# Sample Data
df = pd.DataFrame({
    'Category': ['A', 'B', 'C', 'A', 'B', 'C'],
    'Value': [10, 20, 15, 25, 30, 10]
})

# Create the binding and specify the default value
dropdown = alt.binding_select(options=['A', 'B', 'C'], name='Select Category: ')
my_param = alt.param(name='Category_Filter', value='A', bind=dropdown)

# Build the Chart
chart = alt.Chart(df).mark_bar().encode(
    x='Category:N',
    y='Value:Q',
    color='Category:N'
).add_params(
    my_param
).transform_filter(
    my_param
)

# Display in Streamlit
st.altair_chart(chart, use_container_width=True)