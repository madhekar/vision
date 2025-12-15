import altair as alt
from vega_datasets import data
import pandas as pd
import streamlit as st

print(alt.__version__)
cars = data.cars()
print(cars.head(5))

def plot_cars():
    ch = alt.Chart(cars).mark_bar().encode(
        alt.X('Horsepower', axis=alt.Axis(title="HORSEPOWER")),
        alt.Y('Miles_per_Gallon', axis=alt.Axis(title="Miles Per Gallon")),
        color='Origin',
        shape='Origin'
    ).configure_axis(
        labelFontSize=20,
        titleFontSize=20
    )
    st.altair_chart(ch)

def plot_scale_cars():
    # "Fold" the data to combine multiple columns into two: 'key' (variable name) and 'value'
    df_melted = cars.melt(id_vars=['Origin'], value_vars=['Horsepower', 'Acceleration', 'Miles_per_Gallon'])

    chart = alt.Chart(df_melted).mark_bar().encode(
        x='Origin:N',
        y=alt.Y('value:Q', title='Value'),
        color='Origin:N',
        column='key:N' # Facet by the variable name
    ).resolve_scale(
        y='independent' # Make the y-scales independent for each facet
    ).properties(
        title='Faceted Charts with Independent Y-Scales'
    )    

    st.altair_chart(chart)

def plot_dual_axis():

    # Example data
    data = pd.DataFrame({
        'x': range(10),
        'y1': [1, 5, 2, 8, 3, 9, 4, 7, 5, 10], # Scale ~1-10
        'y2': [100, 500, 200, 800, 300, 900, 400, 700, 500, 1000] # Scale ~100-1000
    })

    base = alt.Chart(data)

    line1 = base.mark_line(color='blue').encode(
        x='x',
        y='y1',
    )

    line2 = base.mark_line(color='red').encode(
        x='x',
        y=alt.Y('y2', axis=alt.Axis(orient='right')), # Right axis
    )

    chart = (line1 + line2).resolve_scale(
        y='independent' # Ensure each layer gets an independent scale
    ).properties(
        title='Dual Axis Chart (Use with caution)'
    )

    st.altair_chart(chart)

def plot_play_scales():
    base = alt.Chart(cars).mark_point().encode(
    x='Horsepower:Q',
    y='Miles_per_Gallon:Q'
    ).properties(
        width=500,
        height=500
    )
    alt.concat(
        base.encode(color='Origin:N'),
        base.encode(color='Cylinders:O')
    ).resolve_scale(
        color='independent'
    )

    st.altair_chart(base)

def plot_inter_1():
   slider = alt.binding_range(min=0, max=1, step=0.05, name='opacity:')
   op_var = alt.param(value=0.1, bind=slider)
   #op_var = alt.param(value=0.1)

   ch= alt.Chart(cars).mark_circle(opacity=op_var).encode(
    x='Horsepower:Q',
    y='Miles_per_Gallon:Q',
    color='Origin:N'
   ).add_params(
       op_var
   )   

   st.altair_chart(ch) 
#plot_scale_cars()
#plot_dual_axis()
#plot_play_scales()

plot_inter_1()