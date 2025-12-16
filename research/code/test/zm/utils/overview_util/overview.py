import os
import pandas as pd
import streamlit as st
import altair as alt
from utils.util import storage_stat as ss
from utils.config_util import config
from utils.util import model_util as mu


# https://www.color-hex.com/color-palette/164
colors = ['#6d765b','#A5BFA6']#['#847577','#cfd2cd']#['#f07162','#0081a7']#['#f97171','#8ad6cc']
#["#ae5a41", "#1b85b8"]#["#636B2F","#BAC095"] #["#9EB8A0", "#58855c"]#['#58855c','#0D3311']#["#BAC095", "#636B2F"]

def extract_folder_paths():
    raw_data_path, input_data_path, app_data_path, final_data_path = (
        config.overview_config_load()
    )
    return (raw_data_path, input_data_path, app_data_path, final_data_path)

def display_storage_metrics(tm, um, fm):
    c1, c2, c3 = st.columns([1.0, 1.0, 1.0])
    with c1:
        #st.markdown(''' ##### :orange[**DISK USAGE**]''' )
        st.markdown("""##### <span style='color:#2d4202'><u>**DISC USAGE**</u></span>""",unsafe_allow_html=True)
        mem = pd.DataFrame({
            'memory': ['Total', 'Used', 'Free'],
            'size': [tm, um, fm]
        })
        ch_mem = alt.Chart(mem).mark_arc(innerRadius=30).encode(
            theta="size",
            color='memory:N'
        ) #.properties(title="Memory Disk Usage Status (GB)")
        st.altair_chart(ch_mem)
        #st.metric(label="TOTAL DISK SIZE (GB)", delta_color="inverse", value=tm, delta=0.1)
    # with c2:
    #     st.metric(label="USED DISK SIZE (GB)", delta_color="inverse", value=um, delta=0.1)
    # with c3:
    #     st.metric(label="FREE DISK SIZE (GB)",delta_color="inverse", value=fm, delta=0.1)

def display_folder_details(dfi, dfv, dfd, dfa, dfn):
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
    with c1:
        dfi = dfi.reset_index()
        ch_count = (
            alt.Chart(dfi)
            .mark_bar()
            .encode(
                x=alt.X("count:Q", title="Images Count"),
                y=alt.Y("index:N", title="Image Type", sort="y"),
                color=alt.Color("index:N"),
                tooltip=["index", "count"],
            )
            .properties(title="Images Count")
        )
  
        ch_size = (
            alt.Chart(dfi)
            .mark_bar()
            .encode(
                x=alt.X("size:Q", title="Images Size"),
                y=alt.Y("index:N", title="Image Type", sort="y"),
                color=alt.Color("index:N"),
                tooltip=["index", "size"],
            )
            .properties(title="Images Size")
        )
        st.altair_chart(ch_count & ch_size)
    with c2:
        dfv = dfv.reset_index()
        ch_count = (
            alt.Chart(dfv)
            .mark_bar()
            .encode(
                x=alt.X("count:Q", title="Videos Count"),
                y=alt.Y("index:N", title="Video Type", sort="y"),
                color=alt.Color("index:N"),
                tooltip=["index", "count"],
            )
            .properties(title="Videos Count")
        )

        ch_size = (
            alt.Chart(dfv)
            .mark_bar()
            .encode(
                x=alt.X("size:Q", title="Videos Size"),
                y=alt.Y("index:N", title="Video Type", sort="y"),
                color=alt.Color("index:N"),
                tooltip=["index", "size"],
            )
            .properties(title="Videos Size")
        )
        st.altair_chart(ch_count & ch_size)

    with c3:
        dfd = dfd.reset_index()
        ch_count = (
            alt.Chart(dfd)
            .mark_bar()
            .encode(
                x=alt.X("count:Q", title="Documents Count"),
                y=alt.Y("index:N", title="Document Type", sort="y"),
                color=alt.Color("index:N"),
                tooltip=["index", "count"],
            )
            .properties(title="Documents Count")
        )

        ch_size = (
            alt.Chart(dfd)
            .mark_bar()
            .encode(
                x=alt.X("size:Q", title="Documents Size"),
                y=alt.Y("index:N", title="Document Type", sort="y"),
                color=alt.Color("index:N"),
                tooltip=["index", "size"],
            )
            .properties(title="Documents Size")
        )
        st.altair_chart(ch_count & ch_size)
    with c4:
        dfa = dfa.reset_index()
        ch_count = (
            alt.Chart(dfa)
            .mark_bar()
            .encode(
                x=alt.X("count:Q", title="Audios Count"),
                y=alt.Y("index:N", title= "Audio Type", sort="y"),
                color=alt.Color("index:N"),
                tooltip=["index", "count"],
            )
            .properties(title="Audios Count")
        )

        ch_size = (
            alt.Chart(dfa)
            .mark_bar()
            .encode(
                x=alt.X("size:Q", title="Audios Size"),
                y=alt.Y("index:N", title="Audio Type", sort="y"),
                color=alt.Color("index:N"),
                tooltip=["index", "size"],
            )
            .properties(title="Audio Size")
        )
        st.altair_chart(ch_count & ch_size)

    with c5:
        dfn = dfn.reset_index()
        ch_count = (
            alt.Chart(dfn)
            .mark_bar()
            .encode(
                x=alt.X("count:Q", title="Other File Count"),
                y=alt.Y("index:N", title="Other File Type", sort="y"),
                color=alt.Color("index:N"),
                tooltip=["index", "count"],
            )
            .properties(title="Other File Count")
        )

        ch_size = (
            alt.Chart(dfn)
            .mark_bar()
            .encode(
                x=alt.X("size:Q", title="Other File Size"),
                y=alt.Y("index:N", title="Other File Type", sort="y"),
                color=alt.Color("index:N"),
                tooltip=["index", "size"],
            )
            .properties(title="Other File Size")
        )
        st.altair_chart(ch_count & ch_size)

def execute():
    rdp, idp, adp, fdp = extract_folder_paths() 
    print(f'--> {rdp}')
    c1, c2 = st.columns([.1,.9])
   
    with c1:
        efs = mu.extract_user_raw_data_folders(rdp)
        #st.markdown('##### :blue[**DATA SOURCES**]')
        st.markdown("""##### <span style='color:#2d4202'><u>**DATA SOURCES**</u></span>""",unsafe_allow_html=True)
        #with st.container(height=100, border=False):        
        #st.markdown('<div class="scrollable-div">', unsafe_allow_html=True)
        with st.container(height=100, border=False):
            for ds in efs:
                st.markdown(f'##### :green[**{ds}**]')
        #st.markdown('</div>', unsafe_allow_html=True)   
        #st.text_area(label="Data Sources", value=efs)
    with c2:
       display_storage_metrics(*ss.extract_server_stats())

  
    st.divider()

    #st.markdown('''##### :blue[RAW DATA]''') 
    st.markdown("""##### <span style='color:#2d4202'><u>**RAW DATA**</u></span>""",unsafe_allow_html=True)
    #st.divider()
    display_folder_details(*ss.extract_all_folder_stats(rdp))

    #st.markdown("##### :blue[**INPUT DATA**]")
    st.markdown("""##### <span style='color:#2d4202'><u>**INPUT DATA**</u></span>""",unsafe_allow_html=True)
    #st.divider()
    display_folder_details(*ss.extract_all_folder_stats(idp))

    st.markdown("""##### <span style='color:#2d4202'><u>**APP DATA**</u></span>""",unsafe_allow_html=True)
    #st.markdown("##### :blue[**APP DATA**]")
    #st.divider()
    display_folder_details(*ss.extract_all_folder_stats(adp))

    st.markdown("""##### <span style='color:#2d4202'><u>**FINAL DATA**</u></span>""",unsafe_allow_html=True)
    #st.markdown("##### :blue[**FINAL DATA**]")
    #st.divider()
    display_folder_details(*ss.extract_all_folder_stats(fdp))

# if __name__ == "__main__":
#     execute()