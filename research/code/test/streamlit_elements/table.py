import streamlit as st
import pandas as pd
import numpy as np

df1 = pd.DataFrame(np.random.randn(5, 5), columns=("col %d" % i for i in range(5)))

my_table = st.table(df1)

df2 = pd.DataFrame(np.random.randn(5, 5), columns=("col %d" % i for i in range(5)))

#my_table.add_rows(df2)


my_chart = st.line_chart(df1)

#my_chart.add_rows(df2)

c=st.button('add row')

if c:
    my_table.add_rows(df2)
    my_chart.add_rows(df2)