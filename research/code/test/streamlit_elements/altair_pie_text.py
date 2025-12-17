import pandas as pd
import altair as alt
import streamlit as st

st.title("Streamlit Altair Pie Chart with Text Labels")

def pie_1():
    # Sample Data
    source = pd.DataFrame({
        "category": ["A", "B", "C", "D", "E", "F"],
        "value": [4, 6, 10, 3, 7, 8]
    })

    # Create the base chart
    # Encode the 'value' field as the theta (angle) and 'category' for color
    base = alt.Chart(source).encode(
        alt.Theta("value:Q").stack(True),
        alt.Color("category:N").legend(None) # Hide the default legend for a cleaner look
    )

    # Layer 1: The pie/arc segments
    pie = base.mark_arc(outerRadius=120, innerRadius=60).encode( # InnerRadius creates a donut chart
        # You can add tooltips for interactivity
        tooltip=["category:N", "value:Q"]
    )

    # Layer 2: The text labels
    text = base.mark_text(radius=140, size=15).encode(
        text="category:N", # Use the 'category' field for the text
        order=alt.Order("value:Q", sort="descending") # Optional: ensure labels match segment order
    )

    # Combine the two layers and display in Streamlit
    chart = pie + text
    st.altair_chart(chart, use_container_width=True)

def pie_2():


    source = pd.DataFrame({
        "category": ["A", "B", "C", "D", "E"],
        "value": [4, 6, 10, 3, 7]
    })

    # Base chart creation
    # Use stack=True for the theta encoding to position labels correctly
    base = alt.Chart(source).encode(
        theta=alt.Theta("value:Q", stack=True),
        color=alt.Color("category:N", legend=None)
    )

    # Pie chart layer (the slices)
    pie = base.mark_arc(outerRadius=120).encode(
        # Add tooltip for interactivity
        tooltip=["category:N", "value:Q"]
    )

    # Text layer (the numeric labels inside the slices)
    text = base.mark_text(radius=80, size=15).encode(
        # Encode the 'value' field as the text
        text="value:Q",
        # Optional: add color encoding for better contrast
        color=alt.value("white") 
    )

    # Combine the pie chart and text layers
    final_chart = pie + text

    # Display the chart in Streamlit
    st.altair_chart(final_chart, use_container_width=True)


def pie_3():
    # 1. Create a sample DataFrame
    source = pd.DataFrame({
        "category": ["A", "B", "C", "D", "E"],
        "value": [4, 6, 10, 3, 7]
    })

    # 2. Create the combined label field
    # This formats the value as an integer for cleaner presentation in the legend/tooltip
    source['legend_label'] = source['category'] + ' (' + source['value'].astype(str) + ')'

    # 3. Create the base chart
    # Encode theta by the value, and color by the new combined label
    base = alt.Chart(source).encode(
        theta=alt.Theta("value:Q", stack=True)
    )

    # 4. Create the pie (arc) layer
    pie = base.mark_arc(outerRadius=120).encode(
        color=alt.Color("legend_label:N", legend=alt.Legend(title="Category (Value)")),
        # Add tooltip for better interactivity, using the combined label field
        tooltip=["category:N", "value:Q", alt.Tooltip("legend_label:N", title="Category (Value)")]
    )

    # # 5. (Optional) Add text labels inside or outside the chart
    # text = base.mark_text(radius=140).encode(
    #     text=alt.Text("value:Q"),
    #     color=alt.value("black")
    # )

    # 6. Combine the layers and display in Streamlit
    chart = pie #+ text
    st.altair_chart(chart, use_container_width=True)

pie_3()    