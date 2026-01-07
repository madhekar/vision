import altair as alt
import pandas as pd
import streamlit as st

# Sample Data (replace with your actual data source)
source = pd.DataFrame(
    {
        "Origin": ["USA", "USA", "Europe", "Japan", "USA", "Europe"],
        "Horsepower": [150, 120, 100, 90, 110, 130],
    }
)

chart = (
    alt.Chart(source)
    .transform_window(
        # The 'op' is implied as 'rank' by using the rank() function in the field definition
        rank_Origin="rank()",
        groupby=[
            "Origin"
        ],  # Optional: groups the data before ranking within each group
        sort=[alt.SortField("Horsepower", order="descending")],
    )
    .mark_point()
    .encode(x="Horsepower:Q", y="rank_Origin:Q", color="Origin:N")
    .properties(title="Rank of Horsepower within Origin")
)

# Display the chart
st.altair_chart(chart)
