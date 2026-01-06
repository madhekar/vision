import pandas as pd
import altair as alt
import streamlit as st


def create_kpi_chart():
    # 1. Prepare your KPI data in a pandas DataFrame
    source = pd.DataFrame(
        {
            "Metric": ["Revenue", "Profit", "Customers", "Satisfaction"],
            "Actual": [50000, 15000, 1200, 4.5],
            "Target": [45000, 14000, 1000, 4.0],
        }
    )

    # 2. Create the base chart
    base = (
        alt.Chart(source)
        .encode(y=alt.Y("Metric", title="KPI"), tooltip=["Metric", "Actual", "Target"])
        .properties(title="KPI Performance")
    )

    # 3. Add actual values as bars
    bars = base.mark_bar(color="blue", opacity=0.7).encode(
        x=alt.X("Actual", title="Value")
    )

    # 4. Add target lines for comparison
    targets = base.mark_tick(color="red", thickness=2, size=20).encode(x="Target")

    # 5. Combine the layers and make it interactive
    chart = bars + targets

    # Display the chart (works in

    st.altair_chart(chart)

create_kpi_chart()    