# Source - https://stackoverflow.com/a/73522095
# Posted by joelostblom
# Retrieved 2026-01-12, License - CC BY-SA 4.0

import altair as alt
from vega_datasets import data


source = data.barley()

n_top_sites = 3

alt.Chart(source).mark_bar().encode(
    x='mean_yield_per_site:Q',
    y='year:O',
    color='year:N',
).facet(
    alt.Facet('site', sort=['mean_yield_per_site'], title=''),
    columns=1,
).transform_joinaggregate(
    mean_yield_per_site='mean(yield)',
    groupby=['site']
).transform_window(
    rank='dense_rank()',  # Don't skip any rank numbers, https://vega.github.io/vega-lite/docs/window.html#ops
    sort=[alt.SortField('mean_yield_per_site', order='descending')]
).transform_filter(
    f'datum.rank <= {n_top_sites}'
)
