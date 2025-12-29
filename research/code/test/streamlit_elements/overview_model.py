import pandas as pd
import altair as alt
import streamlit as st

df =  pd.DataFrame([
{"file_type":".jpg"   ,"count":294.0   ,"size":167.03     ,"category":"image"},
{"file_type":".jpeg"    ,"count":3.0     ,"size":0.86     ,"category":"image"},
{"file_type":".png"   ,"count":411.0  ,"size":4450.64     ,"category":"image"},
{"file_type":".csv "   ,"count":12.0     ,"size":0.27  ,"category":"document"},
{"file_type":".json"    ,"count":4.0     ,"size":2.07    ,"category":"other"},
{"file_type":".csv "   ,"count":12.0     ,"size":0.27     ,"category":"other"}
])

print(df.head(10))

# df_long = df.melt(id_vars=['file_type', 'category'],
#                   var_name='data_type',
#                   value_name='value')

# print(df_long)

# c6 = alt.Chart(df_long).mark_circle().encode(
#     x=alt.X('category:N'),
#     y=alt.Y('value:Q'),
#     color=alt.Color("file_type:N"),
#     xOffset=("category:N"),
#     size="value:Q"
# )

# st.altair_chart(c6)

# c1 = alt.Chart(df).mark_bar().encode(
#     x=alt.X('count:Q', bin=alt.Bin(maxbins=10), scale=alt.Scale(domain=(30, 300), reverse=True)),
#     y=alt.Y('size:Q'),
#     color="file_type:N",
#     size="category:N"
# )

# c2 = alt.Chart(df).mark_bar().encode(
#     x= 'count()',
#     y=alt.Y('category:N')
# )

# st.altair_chart(c1 | c2)

c3 = alt.Chart(df).mark_bar().encode(
    x=alt.X("category:N"),
    y=alt.Y('count:Q'),
    xOffset="category:N",
    color="file_type:N",
)

c4 = (
    alt.Chart(df)
    .mark_bar()
    .encode(
        x=alt.X("category:N"),
        y=alt.Y("size:Q", scale=alt.Scale(zero=False)), #bin=alt.Bin(maxbins=30)),
        xOffset="category:N",
        color="file_type:N",
    )
)
st.altair_chart(c3 | c4)