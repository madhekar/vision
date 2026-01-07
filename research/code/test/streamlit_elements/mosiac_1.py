import altair as alt
import streamlit as st

# Example: Loading a dataset and visualizing with a mosaic-like structure
from vega_datasets import data

# Load the cars dataset (example for demonstration)
cars_df = data("cars")

# Create a mosaic-like plot showing origins and cylinders
chart= alt.Chart(cars_df).mark_rect().encode(
    x=alt.X("Origin", title="Country of Origin"),
    y=alt.Y("Cylinders", title="Number of Cylinders"),
    color="Horsepower",  # Color can represent another variable
    tooltip=["Origin", "Cylinders", "Horsepower"],
).properties(title="Mosaic Plot Concept: Car Origins & Cylinders")


# Display the chart
st.altair_chart(chart)
