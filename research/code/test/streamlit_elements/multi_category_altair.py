import altair as alt
import pandas as pd
import streamlit as st
from vega_datasets import data


# Sample data
# data = pd.DataFrame({
#     'Category': ['A', 'A', 'B', 'B', 'C', 'C'],
#     'SubCategory': ['X', 'Y', 'X', 'Y', 'X', 'Y'],
#     'SubSubCat':['N','H',"N","H","N","H"],
#     'Value': [10, 15, 7, 20, 14, 11]
# })

# chart = alt.Chart(data).mark_bar().encode(
#     # Primary category on the X-axis
#     x='Category:N',
#     # Quantitative value on the Y-axis
#     y=alt.Y('sum(Value):Q', title='Sum of Value'),
#     # Sub-category as color
#     color='SubSubCat:N',
#     # Tooltip to show details on hover
#     tooltip=['Category', 'SubCategory','SubSubCat', 'Value']
# ).properties(
#     title='Hierarchical Stacked Bar Chart'
# )



source = data.barley()

chart = alt.Chart(source).mark_bar().encode(
    # X-axis for the sub-category (year), with 'O' for ordinal type
    x=alt.X('year:O', axis=None), 
    # Y-axis for the quantitative value (sum of yield)
    y=alt.Y('sum(yield):Q', title='Total Yield'),
    # Color distinguishes the sub-categories (years) within each group
    color='year:N', 
    # 'column' creates separate panels (facets) for the primary category (site)
    column='site:N' 
).properties(
    title='Barley Yield by Year and Site'
) #.configure_column(
#     # Optional: Configure column headers for better display
#     header=alt.Header(titleOrient="bottom", labelOrient="bottom")
# )

st.altair_chart(chart)