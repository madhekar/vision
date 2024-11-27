import os
import streamlit as st
import pandas as pd
import numpy as np
import util
import altair as alt
from utils.util import storage_stat as ss
from utils.config_util import config
from utils.util import model_util as mu

colors = ["#BAC095", "#636B2F"]
def extract_folder_paths():
    raw_data_path, input_data_path, app_data_path, final_data_path = (
        config.overview_config_load()
    )
    return (raw_data_path, input_data_path, app_data_path, final_data_path)

def display_storage_metrics(tm, um, fm):
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.metric("TOTAL DISK SIZE (GB)", tm, 0.1)
    with c2:
        st.metric("USED DISK SIZE (GB)", um, 0.1)
    with c3:
        st.metric("FREE DISK SIZE (GB)", fm, 0.1)

def display_folder_details(dfi, dfv, dfd, dfa, dfn):
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
    with c1:
        st.bar_chart(
            dfi,
            horizontal=False,
            stack=True,
            y_label='total size & count of images',
            use_container_width=True,
            color=colors
        )
        # ch = alt.Chart(dfi).mark_bar(opacity=.8).encode(x="type", y="count")
        # ch1 = alt.Chart(dfi).mark_bar(opacity=.5).encode(x="type", y="size")
        # c = alt.layer(ch, ch1)
        # st.altair_chart(c)
    with c2:
        st.bar_chart(
            dfv,
            horizontal=False,
            stack=True,
            y_label="total size & count of videos",
            use_container_width=True,
            color=colors
        )
    with c3:
        st.bar_chart(
            dfd,
            horizontal=False,
            stack=True,
            y_label="total size & count of documents",
            use_container_width=True,
            color=colors
        )
    with c4:
        st.bar_chart(
            dfa,
            horizontal=False,
            stack=True,
            y_label="total size & count of audios",
            use_container_width=True,
            color=colors
        )
    with c5:
        st.bar_chart(
            dfn,
            horizontal=False,
            stack=True,
            y_label="total size & count of other files",
            use_container_width=True,
            color=colors
        )

def execute():
    rdp, idp, adp, fdp = extract_folder_paths() 

    c1, c2 = st.columns([.25,.75])
   
    with c1:
        efs = mu.extract_user_raw_data_folders(rdp)
        st.caption('**AVAILABLE DATA SOURCES**')
        for ds in efs:
            st.write(f'**{ds}**')
        #st.text_area(label="Data Sources", value=efs)
    with c2:
       display_storage_metrics(*ss.extract_server_stats())

    st.subheader("STORAGE OVERVIEW", divider="gray")

    st.caption('**RAW DATA** FOLDER DETAILS')
    display_folder_details(*ss.extract_all_folder_stats(rdp))

    st.caption("**INPUT DATA** FOLDER DETAILS")
    display_folder_details(*ss.extract_all_folder_stats(idp))

    st.caption("**APP DATA** FOLDER DETAILS")
    display_folder_details(*ss.extract_all_folder_stats(adp))

    st.caption("**FINAL DATA** FOLDER DETAILS")
    display_folder_details(*ss.extract_all_folder_stats(fdp))



# st.subheader("STORAGE DETAIL", divider="gray")

# df1 = pd.DataFrame(np.random.randn(10, 5), columns=("col %d" % i for i in range(5)))

# my_table = st.table(df1)

# df2 = pd.DataFrame(np.random.randn(10, 5), columns=("col %d" % i for i in range(5)))

# my_table.add_rows(df2)
# # Now the table shown in the Streamlit app contains the data for
# # df1 followed by the data for df2.

# # Assuming df1 and df2 from the example above still exist...
# my_chart = st.line_chart(df1)
# my_chart.add_rows(df2)
# # Now the chart shown in the Streamlit app contains the data for
# # df1 followed by the data for df2.

# my_chart = st.vega_lite_chart(
#     {
#         "mark": "line",
#         "encoding": {"x": "a", "y": "b"},
#         "datasets": {
#             "some_fancy_name": df1,  # <-- named dataset
#         },
#         "data": {"name": "some_fancy_name"},
#     }
# )
# my_chart.add_rows(some_fancy_name=df2)  # <-- name used as keyword

# """
#   raw_data_path: '/home/madhekar/work/home-media-app/data/raw-data'
#   input_data_path: '/home/madhekar/work/home-media-app/data/input-data'
#   app_data_path: '/home/madhekar/work/home-media-app/data/app-data'
#   final_data_path: '/home/madhekar/work/home-media-app/data/final-data'  

# """



# if __name__ == "__main__":
#     execute()