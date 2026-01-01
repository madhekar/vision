# Source - https://stackoverflow.com/q/66106111
# Posted by viydanofyi, modified by community. See post 'Timeline' for change history
# Retrieved 2026-01-01, License - CC BY-SA 4.0
import altair as alt
import pandas as pd
import streamlit as st
fruits = ['Apples', 'Pears', 'Nectarines', 'Plums', 'Grapes', 'Strawberries']
years = ["2015", "2016", "2017"]

exports = {'fruits' : fruits,
           '2015'   : [2, 1, 4, 3, 2, 4],
           '2016'   : [5, 3, 4, 2, 4, 6],
           '2017'   : [3, 2, 4, 4, 5, 3]}
imports = {'fruits' : fruits,
           '2015'   : [-1, 0, -1, -3, -2, -1],
           '2016'   : [-2, -1, -3, -1, -2, -2],
           '2017'   : [-1, -2, -1, 0, -2, -2]}

df_exp = pd.DataFrame(exports)
df_imp = pd.DataFrame(imports)


cols_year_imp = df_imp.columns[1:]
cols_year_exp = df_exp.columns[1:]

imp = alt.Chart(df_imp).transform_fold(
    list(cols_year_imp)
).mark_bar(
    tooltip=True
).encode(
    x='value:Q',
    y='fruits:N',
    color=alt.Color('key:O', scale=alt.Scale(scheme='reds'))
)

exp = alt.Chart(df_exp).transform_fold(
    list(cols_year_exp)
).mark_bar(
    tooltip=True
).encode(
    x=alt.X('value:Q',title="Export"),
    y='fruits:N',
    color=alt.Color('key:O', scale=alt.Scale(scheme='independant', reverse=True)),
    order=alt.Order('key:O', sort='ascending')
)

# imp | exp
# imp
# exp
# alt.hconcat(imp, exp)
st.altair_chart(imp | exp)