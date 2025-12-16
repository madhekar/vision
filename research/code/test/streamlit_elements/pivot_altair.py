import pandas as pd
import altair as alt
import numpy as np
import streamlit as st

# 1. Create a sample 'complex' dataframe
data = {
    'Region': ['North', 'North', 'South', 'South', 'East', 'East'],
    'Product': ['A', 'B', 'A', 'B', 'A', 'B'],
    'Sales': [100, 150, 80, 120, 200, 180]
}
df = pd.DataFrame(data)

# 2. Create a pivot table (e.g., total sales by region)
# We use aggfunc='sum' and reset_index() to turn the pivot table into a flat DataFrame
pivot_df = df.pivot_table(
    index='Region',
    values='Sales',
    aggfunc='sum'
).reset_index()

print(pivot_df)
# Output:
#   Region  Sales
# 0   East    380
# 1  North    250
# 2  South    200

# 3. Create the Altair pie chart
base = alt.Chart(pivot_df).encode(
    theta=alt.Theta("Sales:Q", stack=True), # Quantitative data type for angle
    color=alt.Color("Region:N"),          # Nominal data type for color categories
)

pie_chart = base.mark_arc(outerRadius=120).properties(
    title='Total Sales by Region'
)

# Optional: Add text labels
text_labels = base.mark_text(radius=140).encode(
    text=alt.Text("Sales:Q"), # Display the exact sales value
    order=alt.Order("Sales:Q", sort="descending"),
    color=alt.value("black")
)

# Combine the pie chart and labels
final_chart = pie_chart + text_labels

# Display the chart (works in Jupyter, Colab, or Streamlit environments)
st.altair_chart(final_chart)
