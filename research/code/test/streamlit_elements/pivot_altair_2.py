import pandas as pd
import altair as alt
import numpy as np
import streamlit as st

# 1. Create sample data and pivot table with multi-index
data = {
    'Category1': ['A', 'A', 'B', 'B', 'A', 'B', 'A', 'B'],
    'Category2': ['X', 'Y', 'X', 'Y', 'X', 'Y', 'Y', 'X'],
    'Value': np.random.randint(1, 20, 8)
}
df = pd.DataFrame(data)

# Create a pivot table (resulting in a multi-level index Series)
# We'll use a simple aggregation, e.g., sum
pivot_table = df.pivot_table(index=['Category1', 'Category2'], values='Value', aggfunc='sum')

print("Original Multi-Index Pivot Table:")
print(pivot_table)
print("-" * 30)

# 2. Flatten the multi-index to columns
# The indices become regular columns named 'Category1' and 'Category2'
source = pivot_table.reset_index()

print("Flattened DataFrame:")
print(source)
print("-" * 30)

# 3. Create the Altair pie chart
# We use 'mark_arc' for pie charts.

base = alt.Chart(source).encode(
    # Use the 'Value' column for the angle/theta encoding
    theta=alt.Theta("Value", stack=True),
    # Combine the two category columns for a distinct color for each slice
    color=alt.Color("Category1:N", title="Category 1"),
    opacity=alt.Opacity("Category2:N", title="Category 2"), # Optional: use the second level for something like opacity
    order=alt.Order("Value", sort="descending") # Sort slices by value
)

# Create the main arc marks (the "pie" slices)
pie = base.mark_arc(outerRadius=120).encode(
    tooltip=['Category1', 'Category2', 'Value'] # Add tooltips for better info
)

# Optional: Add text labels
text = base.mark_text(radius=140).encode(
    text='Value', # Display the value as text
    order=alt.Order("Value", sort="descending"),
    color=alt.value('black')
)

# Combine the layers
chart = pie + text

# Display the chart (if in a Jupyter environment)
st.altair_chart(chart)
