import altair as alt
import pandas as pd

# 1. Sample Data
data = pd.DataFrame({
    'category': ['A', 'B', 'C', 'D'],
    'value': [25, 35, 15, 25]
})

# 2. Create the Pie Chart
pie_chart = alt.Chart(data).mark_arc(outerRadius=120).encode(
    # Arc angle based on 'value'
    theta=alt.Theta("value", stack=True),
    # Color by 'category'
    color=alt.Color("category"),
    # Tooltip for interactivity
    tooltip=["category", "value"]
).properties(
    title="Interactive Pie Chart"
)

# 3. Add Basic Interactivity (Pan/Zoom/Tooltip)
interactive_pie = pie_chart.interactive()

# 4. (Optional) Add a Selection for Linked Actions (e.g., filter a table)
selector = alt.selection_point(fields=['category'], on='click')
pie_chart = pie_chart.add_params(selector)
# ... then use 'selector' in a linked table chart

# Display the chart (in Jupyter/Notebook environment)
interactive_pie