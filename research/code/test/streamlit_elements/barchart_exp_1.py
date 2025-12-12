import streamlit as st
import pandas as pd
import plotly.express as px

# Sample data
df = pd.DataFrame({
    "Category": ["A", "B", "C", "D"],
    "Value 1": [10, 20, 15, 25],
    "Value 2": [12, 18, 17, 22]
})

# Melt the DataFrame to have one 'value' column and one 'series' column for the legend
df_melted = df.melt(id_vars=["Category"], var_name="Series", value_name="Value")

# Create a bar chart with Plotly Express
fig = px.bar(df_melted, x="Category", y="Value", color="Series",
             title="Bar Chart with Styled Legend")

# Update layout to style the legend font
fig.update_layout(
    legend=dict(
        font=dict(
            family="Arial",
            size=14,
            color="DarkRed"
            #weight="bold" # Set to "bold" or "normal" (thin) as needed
        )
    ),
    # You can also make the title bold
    title=dict(
        text="<b>Bar Chart with Styled Legend</b>",
        x=0.5, # Center title
        font=dict(size=20)
    )
)

# Display the chart in Streamlit
st.plotly_chart(fig, use_container_width=True)
