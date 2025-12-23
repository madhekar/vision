import altair as alt
from vega_datasets import data
import pandas as pd
import streamlit as st


def filter_1():
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

def filter_2():
    # Sample data
    source = pd.DataFrame([
        {"a": "A", "b": 28},
        {"a": "B", "b": 55},
        {"a": "C", "b": 43},
        {"a": "D", "b": 91},
        {"a": "E", "b": 81},
        {"a": "F", "b": 53},
        {"a": "G", "b": 19},
        {"a": "H", "b": 87},
        {"a": "I", "b": 52},
    ])

    # Define the selection_point
    # 'fields' specifies the data field to select on, 'on' specifies the event
    select = alt.selection_point(fields=['a'], on="click")

    # Create the base chart
    chart = alt.Chart(source, height=200).mark_bar().encode(
        x="a:O",
        y="b:Q",
        # Use a condition to control the opacity based on the selection
        opacity=alt.condition(select, alt.value(1), alt.value(0.3)),
        # Add a stroke to the selected bars for better visibility
        stroke=alt.condition(select, alt.value("black"), alt.value(None)),
        # Adjust stroke width
        strokeWidth=alt.condition(select, alt.value(2), alt.value(0))
    ).add_params(
        # Add the selection parameter to the chart
        select
    )

    st.altair_chart(chart)

def filter_3():


    # Load the data
    source = data.barley()

    # Define the selection_point
    # encodings=['color'] means clicking on a color in the legend (if shown) or the bar itself will trigger the selection
    # 'toggle=True' allows multiple selections with shift+click
    selector = alt.selection_point(fields=['site', 'year'], bind='legend', name="selection", toggle=True)

    # Create the grouped bar chart
    chart = alt.Chart(source).mark_bar().encode(
        # 'year' defines the groups (outer category on x-axis)
        x=alt.X('year:O', axis=None), # Hide axis for cleaner look since it's in the column headers

        # 'site' defines the sub-categories (individual bars within each group)
        column=alt.Column('site:N', header=alt.Header(titleOrient="bottom", labelOrient="bottom")),

        y=alt.Y('sum(yield):Q', title='Total Yield'),

        # Color is determined by the site, with a condition for interactivity
        color=alt.condition(
            selector,
            'site:N', # Use 'site' color if selected
            alt.value('lightgray') # Gray out if not selected
        ),
        tooltip=['year', 'site', 'sum(yield)']
    ).add_params(
        selector
    ).properties(
        title='Barley Yield by Year and Site with Interactive Selection'
    )

    # Optional: You can layer text labels on top of the bars if needed
    # text = chart.mark_text(align='center', baseline='middle', dy=-5).encode(
    #     text='sum(yield):Q'
    # )
    # chart = chart + text

    st.altair_chart(chart)


def filter_4():

    cars = data.cars()
    print(cars)
    # 1. Define the first dropdown selection
    origin_selection = alt.selection_point(fields=['Origin'], bind='legend', name="Origin")
    # For a dropdown *widget*, use bind=alt.binding_select(options=...)
    origin_dropdown = alt.binding_select(options=cars['Origin'].unique().tolist(), name='Select Origin ')
    origin_selection = alt.selection_point(fields=['Origin'], bind=origin_dropdown)

    # 2. Define the second dropdown selection (e.g., for Cylinders)
    cylinders_dropdown = alt.binding_select(options=sorted(cars['Cylinders'].unique().tolist()), name='Select Cylinders ')
    cylinders_selection = alt.selection_point(fields=['Cylinders'], bind=cylinders_dropdown)

    # Create the chart
    chart = alt.Chart(cars).mark_circle().encode(
        x='Horsepower:Q',
        y='Miles_per_Gallon:Q',
        color='Origin:N',
        size='Acceleration:Q',
        tooltip=['Name', 'Origin', 'Cylinders']
    ).add_params(
        origin_selection,
        cylinders_selection
    ).transform_filter(
        # Combine both selections using logical AND
        origin_selection & cylinders_selection
    )

    st.markdown("""
         <style>
            .vega-bind {text-align:right;}
         </style>
        """ , unsafe_allow_html=True)
    st.altair_chart(chart)

def filter_5():

    cars = data.cars()

    # Define a selection that allows for multiple values ('multi' type)
    # The bind='legend' often works, but a custom binding is better for dropdowns
    # For a true multi-select dropdown widget, you need a custom binding
    # which behaves like checkboxes in the UI.

    select_origin = alt.selection_point(fields=['Origin'], bind='legend', name="OriginSelection")
    # The 'bind=\'legend\'' creates checkboxes by default for multi-select, not a traditional dropdown.

    # To get a multi-select dropdown *widget* that looks like a dropdown box but allows multiple selections,
    # Altair/Vega-Lite currently requires a workaround or specific environment (like Altair Panopticon).
    # In standard Altair/Jupyter/Streamlit, the 'multi' selection with 'bind=\'legend\'' is the closest
    # built-in option for checkbox-style multi-selection.

    # A common approach in Altair is to use another chart as a selection element, or use third-party libraries
    # like Panel or Streamlit components for more complex widget behavior.

    # Basic example using a selection to filter a chart:
    selection = alt.selection_point(fields=['Origin'], on="click", toggle=True)

    chart = alt.Chart(cars).mark_point().encode(
        x='Horsepower:Q',
        y='Miles_per_Gallon:Q',
        color=alt.condition(selection, 'Origin:N', alt.value('lightgray')),
        tooltip=['Name', 'Origin', 'Horsepower', 'Miles_per_Gallon']
    ).add_params(
        selection
    ).transform_filter(
        selection
    )

    # You can then use this selection to filter other charts in a linked view.

    st.altair_chart(chart)

def filter_interval_6():

    cars = data.cars()

    interval = alt.selection_interval()
    base = alt.Chart(cars).mark_point().encode(
        y='Horsepower',
        color=alt.condition(interval, 'Origin', alt.value('lightgray')),
        tooltip='Name'
    ).add_selection(
        interval
    )
    hist = alt.Chart(cars).mark_bar().encode(
        x='count()',
        y='Origin',
        color='Origin'
    ).properties(
        width=800,
        height=80
    ).transform_filter(
        interval
    )
    scatter = base.encode(x='Miles_per_Gallon') | base.encode(x='Acceleration')
    st.altair_chart(scatter & hist)

filter_interval_6()    