import streamlit as st
import altair as alt
import pandas as pd

# 1. Prepare Data (Example: Sales by Region & Product)
data = pd.DataFrame({
    'region': ['North', 'North', 'South', 'South', 'East', 'East', 'West', 'West'],
    'product': ['A', 'B', 'A', 'C', 'B', 'C', 'A', 'C'],
    'sales': [200, 150, 300, 100, 250, 120, 180, 90]
})

# 2. Create Hierarchy for Treemap
# We'll nest region and product
hierarchy = alt.Hierarchy('region', 'product')

# 3. Build Altair Treemap Chart
# Altair doesn't have a built-in 'treemap' mark, so we use rect and layer/facet
# A common approach uses a calculated 'size' and 'color'
chart = alt.Chart(data).mark_rect().encode(
    x=alt.X('sum(sales)', stack='normalize', axis=None), # Stack for layout
    y=alt.Y('sum(sales)', stack='normalize', axis=None), # Stack for layout
    color=alt.Color('region', legend=None), # Color by region
    tooltip=['region', 'product', 'sales'] # Show details on hover
).properties(
    title='Sales Treemap by Region & Product'
)

# Streamlit specific: Use st.altair_chart for display
st.altair_chart(chart, use_container_width=True, theme="streamlit") # {Link: Altair-viz https://altair-viz.github.io/} [2, 7]

# Add interactivity (Optional): Rerun app on selection
# For complex interactions like filtering, you might use selections with vega_lite
# st.write("Click on a rectangle to see details (requires specific selection config)")
