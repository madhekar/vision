import altair as alt
from vega_datasets import data
import streamlit as st

# Load the cars data
source = data.cars()

# Define a base chart with a rank transform
base = alt.Chart(source).transform_stack(
    stack="count_",
    groupby=["Origin"],
    as_=["y", "y2"],
    offset="normalize",
    sort=[alt.SortField("Cylinders", "ascending")],
).transform_stack(
    stack="count_",
    groupby=["Cylinders"],
    as_=["x", "x2"],
    offset="normalize",
    sort=[alt.SortField("Origin", "ascending")],
).transform_calculate(
    # Calculate new coordinates for the mosaic tiles
    ny="datum.y + (datum.rank_Cylinders - 1) * datum.distinct_Cylinders * 0.01 / 3",
    ny2="datum.y2 + (datum.rank_Cylinders - 1) * datum.distinct_Cylinders * 0.01 / 3",
    nx="datum.x + (datum.rank_Origin - 1) * 0.01",
    nx2="datum.x2 + (datum.rank_Origin - 1) * 0.01",
    xc="(datum.nx+datum.nx2)/2",
    yc="(datum.ny+datum.ny2)/2",
)

# Create the rectangles (the main part of the mosaic)
rect = base.mark_rect().encode(
    x=alt.X("nx:Q", axis=None),
    x2="nx2",
    y=alt.Y("ny:Q", axis=None),
    y2="ny2",
    color=alt.Color("Origin:N", legend=None),
    opacity=alt.Opacity("Cylinders:Q", legend=None),
    tooltip=["Origin:N", "Cylinders:Q", "count_"],
)

# Add text labels for the 'Cylinders' variable within the tiles
text = base.mark_text(baseline="middle").encode(
    x=alt.X("xc:Q", axis=None),
    y=alt.Y("yc:Q", title="Cylinders"),
    text="Cylinders:N"
)

# Add labels for the 'Origin' variable at the top
origin_labels = base.mark_text(baseline="middle", align="center").encode(
    x=alt.X("min(xc):Q", axis=alt.Axis(title="Origin", orient="top")),
    color=alt.Color("Origin", legend=None),
    text="Origin",
)

# Combine the layers and configure the chart
mosaic = rect + text

# Final combined chart with shared scale resolution and view configuration
final_chart = (
    (origin_labels & mosaic)
    .resolve_scale(x="shared")
    .configure_view(stroke="")
    .configure_concat(spacing=10)
    .configure_axis(domain=False, ticks=False, labels=False, grid=False)
)

# Display the chart (in a Jupyter notebook or similar environment)
st.altair_chart(final_chart)
