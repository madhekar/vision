import altair as alt
from vega_datasets import data
import streamlit as st

# 1. Load the data
df = data.cars()

print(df)
# Extract unique values for the dropdown list options
origin_options = df['Origin'].unique().tolist()
print(origin_options)
# 2. Define the dropdown input widget using binding
# 'name' sets the label for the dropdown
dropdown = alt.binding_select(options=origin_options, name='Select Origin: ')

# 3. Define a single selection that uses the dropdown binding
# 'fields' specifies which data column to filter on
# 'init' sets an initial value (optional)
selection = alt.selection_point(
    fields=['Origin'],
    bind=dropdown,
    #init={'Origin': origin_options[0]} # Initialize with the first option
)

# 4. Create the base chart
chart = alt.Chart(df).mark_point().encode(
    x='Horsepower:Q',
    y='Miles_per_Gallon:Q',
    color='Origin:N',
    tooltip=['Name', 'Origin', 'Horsepower', 'Miles_per_Gallon']
).add_selection(
    selection # 5. Add the selection to the chart
).transform_filter(
    selection # 6. Use the selection to filter the data
).properties(
    title="Car Data Filtered by Origin"
)

# Display the chart (in a Jupyter environment, just calling the chart variable works)
st.altair_chart(chart)
