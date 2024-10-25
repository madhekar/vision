import streamlit as st
import pandas as pd
import numpy as np

st.header("OVERVIEW", divider="gray")
c1, c2, c3, c4, c5 = st.columns([1,1,1,1,1])
with c1:
  st.metric("RAW STORAGE", 20000, -50)
with c2:
  st.metric("IMAGE STORAGE", 10000, 50)
with c3:
  st.metric("DUPICATE IMAGES", 700, 5)  
with c4:
  st.metric("BAD IMAGES", 8000, 120)
with c5:
  st.metric("MISSING MEADATA IMAGE", 7000, 35)    


st.subheader("STORAGE OVERVIEW", divider="gray")
c1,c2,c3,c4 = st.columns([1,1,1,1])
with c1:
      chart_data = pd.DataFrame(abs(np.random.randn(1, 4)) *100 , columns=["images", "text", "video", "audio"])
      st.bar_chart(chart_data, horizontal=False, stack=False, y_label="number of files", use_container_width=True)
with c2:
      chart_data = pd.DataFrame(abs(np.random.randn(1, 4)) *100 , columns=["images", "text", "video", "audio"])
      st.bar_chart(chart_data, horizontal=False, stack=False, y_label="number of files", use_container_width=True)  
with c3:
      chart_data = pd.DataFrame(abs(np.random.randn(1, 4)) *100 , columns=["images", "text", "video", "audio"])
      st.bar_chart(chart_data, horizontal=False, stack=False, y_label="number of files", use_container_width=True)
with c4:
      chart_data = pd.DataFrame(abs(np.random.randn(1, 4)) *100 , columns=["images", "text", "video", "audio"])
      st.bar_chart(chart_data, horizontal=False, stack=False, y_label="number of files", use_container_width=True)             

st.subheader("STORAGE DETAIL", divider="gray")

df1 = pd.DataFrame(np.random.randn(10, 5), columns=("col %d" % i for i in range(5)))

my_table = st.table(df1)

df2 = pd.DataFrame(np.random.randn(10, 5), columns=("col %d" % i for i in range(5)))

my_table.add_rows(df2)
# Now the table shown in the Streamlit app contains the data for
# df1 followed by the data for df2.

# Assuming df1 and df2 from the example above still exist...
my_chart = st.line_chart(df1)
my_chart.add_rows(df2)
# Now the chart shown in the Streamlit app contains the data for
# df1 followed by the data for df2.

my_chart = st.vega_lite_chart(
    {
        "mark": "line",
        "encoding": {"x": "a", "y": "b"},
        "datasets": {
            "some_fancy_name": df1,  # <-- named dataset
        },
        "data": {"name": "some_fancy_name"},
    }
)
my_chart.add_rows(some_fancy_name=df2)  # <-- name used as keyword