import altair as alt
from vega_datasets import data
import streamlit as st
# Load the data
source = data.cars()

# 1. Define the input widget (a dropdown selection)
# The `options` are populated from the unique values in the 'Origin' column.
# The `name` is a label for the widget.
input_dropdown = alt.binding_select(
    options=['All'] + list(source['Origin'].unique()), 
    name='Select Origin '
)

# 2. Define the selection parameter
# It's a single selection projected over the 'Origin' field, bound to the dropdown widget.
# `init` sets the initial value.
# `empty=True` allows for an "empty" selection state, which displays all data points initially.
selection = alt.selection_point(
    fields=['Origin'], 
    bind=input_dropdown, 
    name='OriginSelection',
    init={'Origin': 'All'},
    empty=True
)


# 3. Create the base chart
chart = alt.Chart(source).mark_point().encode(
    x='Horsepower:Q',
    y='Miles_per_Gallon:Q',
    tooltip=['Name:N', 'Origin:N', 'Horsepower:Q', 'Miles_per_Gallon:Q']
).properties(
    title='Car Data by Origin Selection'
)

# 4. Apply the selection to filter the data
# We use transform_filter to only show points that match the selection parameter.
final_chart = chart.add_params(
    selection
).transform_filter(
    # This expression filters data where Origin matches the selection, 
    # or if the selection is empty/not defined (i.e., 'All' is selected)
    alt.FieldRangePredicate(field='Origin', range=selection.Origin)
)

# Display the chart (this will work in JupyterLab, Google Colab, etc.)
st.altair_chart(final_chart)
