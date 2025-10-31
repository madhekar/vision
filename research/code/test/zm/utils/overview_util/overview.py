import os
import streamlit as st
from utils.util import storage_stat as ss
from utils.config_util import config
from utils.util import model_util as mu
import streamlit_scrollable_textbox as stx

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
        st.metric(label="TOTAL DISK SIZE (GB)", delta_color="inverse", value=tm, delta=0.1)
    with c2:
        st.metric(label="USED DISK SIZE (GB)", delta_color="inverse", value=um, delta=0.1)
    with c3:
        st.metric(label="FREE DISK SIZE (GB)",delta_color="inverse", value=fm, delta=0.1)

def display_folder_details(dfi, dfv, dfd, dfa, dfn):
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
    with c1:
        st.bar_chart(
            dfi,
            horizontal=False,
            stack=True,
            y_label='total size(MB) & count of image files',
            use_container_width=True,
            color=colors
        )
    with c2:
        st.bar_chart(
            dfv,
            horizontal=False,
            stack=True,
            y_label="total size(MB) & count of video files",
            use_container_width=True,
            color=colors,
        )
    with c3:
        st.bar_chart(
            dfd,
            horizontal=False,
            stack=True,
            y_label="total size(MB) & count of document files",
            use_container_width=True,
            color=colors,
        )
    with c4:
        st.bar_chart(
            dfa,
            horizontal=False,
            stack=True,
            y_label="total size(MB) & count of audio files",
            use_container_width=True,
            color=colors,
        )
    with c5:
        st.bar_chart(
            dfn,
            horizontal=False,
            stack=True,
            y_label="total size(MB) & count of other files",
            use_container_width=True,
            color=colors,
        )

def execute():
    rdp, idp, adp, fdp = extract_folder_paths() 
    print(f'--> {rdp}')
    c1, c2 = st.columns([.25,.75])
   
    with c1:
        efs = mu.extract_user_raw_data_folders(rdp)
        st.caption('**AVAILABLE DATA SOURCES**')
        #with st.container(height=100, border=False):        
        #st.markdown('<div class="scrollable-div">', unsafe_allow_html=True)
        with st.container(key="scrollable-div"):
            for ds in efs:
                st.write(f':: {ds}')
        #st.markdown('</div>', unsafe_allow_html=True)   
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

# if __name__ == "__main__":
#     execute()