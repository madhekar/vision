import pandas as pd
import altair as alt
import streamlit as st

# Sample hierarchical data (long format is best for Altair)
# Each row represents a segment in one of the layers
source = pd.DataFrame({
    'category': ['A', 'B', 'C', 'A', 'B', 'C', 'A1', 'A2', 'B1', 'B2', 'C1', 'C2'],
    'parent':   ['', '', '', 'A', 'B', 'C', 'A', 'A', 'B', 'B', 'C', 'C'],
    'value':    [0, 0, 0, 4, 6, 10, 2, 2, 3, 3, 5, 5],
    'level':    [1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3]
})

# Filter data for each level and create separate charts
# Level 1 (Center) - The total value would ideally be a single point or a calculation
# This example uses placeholder values for level 1 to show the structure
chart1 = alt.Chart(source[source['level'] == 1]).mark_arc(outerRadius=50, innerRadius=0).encode(
    theta=alt.Theta("value:Q"),
    color=alt.Color("category:N", legend=None),
    order=alt.Order("value:Q", sort="descending"),
    tooltip=['category', 'value']
)

# Level 2 (Middle Ring)
chart2 = alt.Chart(source[source['level'] == 2]).mark_arc(outerRadius=100, innerRadius=50).encode(
    theta=alt.Theta("value:Q"),
    color=alt.Color("category:N", legend=None),
    order=alt.Order("value:Q", sort="descending"),
    tooltip=['category', 'value']
)

# Level 3 (Outer Ring) - this data would need to sum up correctly to the parent level for a true sunburst
# This example uses the 'parent' to group colors but 'category' to define slices
chart3 = alt.Chart(source[source['level'] == 3]).mark_arc(outerRadius=150, innerRadius=100).encode(
    theta=alt.Theta("value:Q"),
    # Use parent for general color scheme, category for individual slices
    color=alt.Color("parent:N", scale=alt.Scale(range='category')), 
    order=alt.Order("category:N"),
    tooltip=['category', 'value', 'parent']
)

# Layer the charts using the '+' operator
# The charts are overlaid on the same set of axes
multilevel_pie_chart = (chart1 + chart2 + chart3).properties(
    title="Multilevel Pie Chart (Sunburst) Example"
)

# Display the chart
st.altair_chart(multilevel_pie_chart)