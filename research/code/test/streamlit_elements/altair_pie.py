import pandas as pd
import altair as alt
import streamlit as st

# Example Data (replace with your actual data)
source = pd.DataFrame({
    'category': ['A', 'A', 'B', 'B', 'C', 'C'],
    'subcategory': ['A1', 'A2', 'B1', 'B2', 'C1', 'C2'],
    'value': [10, 20, 15, 25, 5, 25]
})

# Inner Chart (Main Categories)
inner_chart = alt.Chart(source).mark_arc(innerRadius=0, outerRadius=70).encode(
    theta=alt.Theta("value", stack=True),
    color=alt.Color("category"),
    order=alt.Order("category")
)

# Outer Chart (Subcategories)
outer_chart = alt.Chart(source).mark_arc(innerRadius=75, outerRadius=120).encode(
    theta=alt.Theta("value", stack=True),
    color=alt.Color("subcategory"), # Color by subcategory
    order=alt.Order("subcategory")
)

# Layer the charts
nested_pie_chart = inner_chart + outer_chart

# Display the chart (if using a notebook environment)
st.altair_chart(nested_pie_chart) #.show()
