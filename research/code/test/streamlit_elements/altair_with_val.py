import altair as alt
from vega_datasets import data


source = data.wheat()

base = alt.Chart(source).encode(
    x='wheat',
    y="year:O",
    text='wheat'
)
base.mark_bar() + base.mark_text(align='left', dx=2)