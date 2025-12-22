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
'''
data-paths:
  raw_data_path: /data/raw-data
  input_data_path: /data/input-data
  app_data_path: /data/app-data
  final_data_path: /data/final-data
error-paths:
  img_dup_error_path: /data/input-data/error/img/duplicate
  img_qua_error_path: /data/input-data/error/img/quality
  img_mis_error_path: /data/input-data/error/img/missing-data
  video_dup_error_path: /data/input-data/error/video/duplicate
  video_qua_error_path: /data/input-data/error/video/quality
  video_mis_error_path: /data/input-data/error/video/missing-data
  text_dup_error_path: /data/input-data/error/txt/duplicate
  text_qua_error_path: /data/input-data/error/txt/quality
  text_mis_error_path: /data/input-data/error/txt/missing-data
  audio_dup_error_path: /data/input-data/error/audio/duplicate
  audio_qua_error_path: /data/input-data/error/audio/quality
  audio_mis_error_path: /data/input-data/error/audio/missing-data
input-paths:
  image_data_path: /data/input-data/img
  video_data_path: /data/input-data/video
  audio_data_path: /data/input-data/audio
  text_data_path: /data/input-data/txt
final-paths:
  image_data_path: /data/final-data/img
  video_data_path: /data/final-data/video
  audio_data_path: /data/final-data/audio
  text_data_path: /data/final-data/txt
'''
def extract_folder_paths():
    (raw_data_path, input_data_path, app_data_path, final_data_path ,
    img_dup_error_path, img_qua_error_path, img_mis_error_path,
    video_dup_error_path, video_qua_error_path, video_mis_error_path,
    text_dup_error_path, text_qua_error_path, text_mis_error_path,
    audio_dup_error_path, audio_qua_error_path, audio_mis_error_path,
    image_data_path, video_data_path, audio_data_path, text_data_path,
    final_image_data_path, final_video_data_path, final_audio_data_path, final_text_data_path
    ) = (
        config.overview_config_load()
    )
    ovr_path_list = [
        image_data_path, img_dup_error_path, img_qua_error_path, img_mis_error_path,
        video_data_path,video_dup_error_path, video_qua_error_path, video_mis_error_path,
        text_data_path, text_dup_error_path, text_qua_error_path ,text_mis_error_path,
        audio_data_path, audio_dup_error_path, audio_qua_error_path, audio_mis_error_path,
        final_image_data_path, final_video_data_path, final_text_data_path, final_audio_data_path
        ]

    return (raw_data_path, input_data_path, app_data_path, final_data_path,
    ovr_path_list)

def disc_usage(tm, um, fm):

    mem = pd.DataFrame({"disc": ["Total", "Used", "Free"], "size": [tm, um, fm]})

    # This formats the value as an integer for cleaner presentation in the legend/tooltip
    mem["legend_label"] = (mem["disc"] + "  ( " + mem["size"].astype(str) + " GB)")

    # Encode theta by the value, and color by the new combined label
    base = alt.Chart(mem).encode(
        theta=alt.Theta("size:Q", stack=True)
    )
   # 4. Create the pie (arc) layer
    pie = base.mark_arc(outerRadius=120).encode(
        color=alt.Color("legend_label:N", legend=alt.Legend(title="Disc Usage")),
        # Add tooltip for better interactivity, using the combined label field
        tooltip=["disc:N", "size:Q", alt.Tooltip("legend_label:N", title="Disc Usage")]
    )

    st.altair_chart(pie)


def display_storage_metrics(tm, um, fm):
    c1, c2, c3 = st.columns([1.0, 1.0, 1.0])
    with c1:
        st.markdown("""##### <span style='color:#2d4202'><u>**DISC USAGE**</u></span>""",unsafe_allow_html=True)
        disc_usage(tm, um, fm)

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
            .properties(title=f"Images Count Total: {int(dfi['count'].sum())}")
        )
  
        ch_size = (
            alt.Chart(dfi)
            .mark_bar()
            .encode(
                x=alt.X("size:Q", title="Images Size "),
                y=alt.Y("index:N", title="Image Type", sort="y"),
                color=alt.Color("index:N"),
                tooltip=["index", "size"],
            )
            .properties(title=f"Images Size Total: {int(dfi['size'].sum())} GB")
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
            .properties(title=f"Videos Count Total: {int(dfv['count'].sum())}")
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
            .properties(title=f"Videos Size Total: {int(dfv['size'].sum())} GB")
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
            .properties(title=f"Documents Count  Total: {int(dfd['count'].sum())}")
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
            .properties(title=f"Documents Size Total: {int(dfd['size'].sum())} GB")
        )
        st.altair_chart(ch_count & ch_size)
    with c4:
        dfa = dfa.reset_index()
        ch_count = (
            alt.Chart(dfa)
            .mark_bar()
            .encode(
                x=alt.X("count:Q", title="Audios Count"),
                y=alt.Y("index:N", title="Audio Type", sort="y"),
                color=alt.Color("index:N"),
                tooltip=["index", "count"],
            )
            .properties(title=f"Audios Count Total: {int(dfa['count'].sum())}")
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
            .properties(title=f"Audio Size Total: {int(dfa['size'].sum())} GB")
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
            .properties(title=f"Other File Count  Total: {int(dfn['count'].sum())}")
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
            .properties(title=f"Other File Size Total: {int(dfn['size'].sum())} GB")
        )
        st.altair_chart(ch_count & ch_size)

def execute():
    (rdp, idp, adp, fdp, opl) = extract_folder_paths() 

    dfi, dff = ss.acquire_overview_data(opl)

    print(dfi, dff)
    
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