import pandas as pd
import altair as alt
import numpy as np
import streamlit as st

# Example Data Creation (replace with your actual pivot table)
data = {'Category1': ['A', 'A', 'B', 'B'],
        'Category2': ['X', 'Y', 'X', 'Y'],
        'Value1': [10, 15, 12, 18],
        'Value2': [20, 25, 22, 28]}
df_pivot = pd.DataFrame(data)

# Melt the DataFrame to long-form
# 'Category1' and 'Category2' become id_vars
# 'Value1' and 'Value2' are combined into 'Measure' and their values in 'Count'
df_long = df_pivot.melt(id_vars=['Category1', 'Category2'],
                        var_name='Measure',
                        value_name='Count')

print(df_long.head())

# Aggregate data for the outer layer (e.g., total count per Category1)
df_outer = df_long.groupby('Category1')['Count'].sum().reset_index()

# Define the base chart
base = alt.Chart(df_long).encode(
    theta=alt.Theta("Count", stack=True)
)

# Outer Layer (e.g., by Measure or Category1) - customize as needed
# Using 'Measure' for color as an example
outer_pie = base.mark_arc(outerRadius=120, innerRadius=80).encode(
    color=alt.Color("Measure", legend=None),
    order=alt.Order("Count", sort="descending"),
    tooltip=["Measure", "Count"]
)

# Inner Layer (e.g., by Category1)
inner_pie = base.mark_arc(outerRadius=80, innerRadius=40).encode(
    color=alt.Color("Category1"),
    order=alt.Order("Count", sort="descending"),
    tooltip=["Category1", "Count"]
)

# Combine the layers
chart = outer_pie + inner_pie

# Optional: Add text labels (requires further data manipulation/joining for positioning)
# See the Altair gallery for advanced examples with labels:
st.altair_chart(chart)