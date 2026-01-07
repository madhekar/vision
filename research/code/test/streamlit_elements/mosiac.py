import altair as alt
from vega_datasets import data
import streamlit as st
# Load the data
cars = data.cars.url

# Define the base chart with data and common encodings
base = (
    alt.Chart(cars)
    .encode(
        # Use ranking transforms to determine the position within each category
        # The 'Origin' is for the main columns, 'Cylinders' for the sub-rectangles
        alt.X("Origin:N", sort=alt.SortField("Origin", "ascending")),
        alt.Color("Origin:N", legend=None),
    )
    .transform_joinaggregate(
        # Calculate global rank for positioning
        total_count="count()",
        rank_Origin='[{"op": "rank", "field": None, "as": "rank_Origin:O",}]',
        distinct_Origin="distinct(Origin)",
        rank_Cylinders="rank()",
        distinct_Cylinders="distinct(Cylinders)",
    )
    .transform_stack(
        # Stack the 'count_' variable to define the y-axis (heights)
        stack="count_",
        groupby=["Origin"],
        as_=["y", "y2"],
        offset="normalize",  # Normalize to 100% height within each column
        sort=[alt.SortField("Cylinders", "ascending")],
    )
    .transform_calculate(
        # Calculate final positions with minor offsets for visual separation
        ny="datum.y + (datum.rank_Cylinders - 1) * datum.distinct_Cylinders * 0.01 / 3",
        ny2="datum.y2 + (datum.rank_Cylinders - 1) * datum.distinct_Cylinders * 0.01 / 3",
        nx="datum.x + (datum.rank_Origin - 1) * 0.01",
        nx2="datum.x2 + (datum.rank_Origin - 1) * 0.01",
        xc="(datum.nx+datum.nx2)/2",
        yc="(datum.ny+datum.ny2)/2",
    )
)

# Create the main rectangular marks (the mosaic tiles)
rect = base.mark_rect().encode(
    x="nx:Q",  # Use calculated X positions
    x2="nx2",
    y="ny:Q",  # Use calculated Y positions
    y2="ny2",
    color=alt.Color("Origin:N", legend=None),
    opacity=alt.Opacity("Cylinders:Q", legend=None),
    tooltip=["Origin:N", "Cylinders:Q", "count_:Q"],
)

# Add text labels for the 'Cylinders' categories
text = base.mark_text(baseline="middle").encode(
    x="xc:Q", y=alt.Y("yc:Q", title="Cylinders"), text="Cylinders:N"
)

# Combine the rectangles and text layers
mosaic = rect + text

# Add labels for the 'Origin' (X-axis)
origin_labels = base.mark_text(baseline="middle", align="center").encode(
    x=alt.X("min(xc):Q", axis=alt.Axis(title="Origin", orient="top")),
    color=alt.Color("Origin", legend=None),
    text="Origin",
)

# Final assembly and styling
final_chart = (
    (
        origin_labels & mosaic
    )  # Use vertical concatenation (&) to stack origin labels on top
    .resolve_scale(x="shared")
    .configure_view(stroke="")
    .configure_concat(spacing=10)
    .configure_axis(domain=False, ticks=False, labels=False, grid=False)
)

# To display the chart in a Jupyter notebook or similar environment:
# final_chart

st.altair_chart(final_chart)