import altair as alt
import pandas as pd
import streamlit as st

# 1. Create a DataFrame in long format
data = pd.DataFrame({
    'Category': ['Sales', 'Profit', 'Users', 'Revenue'],
    'Value': [12500, 4200, 850, 25000],
    'Unit': ['USD', 'USD', 'Count', 'USD']
})

# 2. Define a base chart template for a single value display
def single_value_chart(df, category_name):
    # Filter the data for the specific category
    df_filtered = df[df['Category'] == category_name]
    
    chart = alt.Chart(df_filtered).mark_text(align='center', baseline='middle').encode(
        text=alt.Text('Value:Q', format='.2s'), # Format quantitative data nicely
        color=alt.value('black')
    ).properties(
        title=category_name, # Use the category as the title
        width=100,
        height=50
    )
    
    # Add the unit label beneath the value
    chart_unit = alt.Chart(df_filtered).mark_text(align='center', baseline='top', dy=20, fontSize=12).encode(
        text='Unit:N',
        color=alt.value('gray')
    )
    
    return chart + chart_unit # Layer the value and unit

# 3. Create a chart for each specific value
chart1 = single_value_chart(data, 'Sales')
chart2 = single_value_chart(data, 'Profit')
chart3 = single_value_chart(data, 'Users')
chart4 = single_value_chart(data, 'Revenue')

# 4. Concatenate the charts horizontally
final_chart = alt.hconcat(chart1, chart2, chart3, chart4).configure_view(
    strokeWidth=0 # Optional: removes the border around each single chart
)

st.altair_chart(final_chart)