import os
import pandas as pd
import streamlit as st
from streamlit_dimensions import st_dimensions
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

    return (raw_data_path, 
            input_data_path, 
            app_data_path, 
            final_data_path,
            ovr_path_list)

def filter_selection(df):
    #print(f'*** {df}')
    interval = alt.selection_interval(encodings=['x','y'])
    # 1. Define the first dropdown selection
    source_selection = alt.selection_point(
        fields=["source"], bind="legend", name="source"
    )
    # For a dropdown *widget*, use bind=alt.binding_select(options=...)
    source_dropdown = alt.binding_select(
        options=sorted(df["source"].unique().tolist()), name="Select source "
    )
    source_selection = alt.selection_point(fields=["source"], bind=source_dropdown)

    # # 2. Define the second dropdown selection
    # data_stage_dropdown = alt.binding_select(
    #     options=sorted(df["data_stage"].unique().tolist()), name="Select data stage "
    # )
    # data_stage_selection = alt.selection_point(
    #     fields=["data_stage"], bind=data_stage_dropdown
    # )

    #3. Define the third selection
    data_type_dropdown = alt.binding_select(
        options=sorted(df["data_type"].unique().tolist()), name="Select data type "
    )
    data_type_selection = alt.selection_point(
        fields=["data_type"], bind=data_type_dropdown
    )

    # Create the chart
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("count:Q", axis=alt.Axis(grid=True, gridColor='grey')),
            y=alt.Y("size:Q", axis=alt.Axis(grid=True, gridColor="grey")),
            size="data_type:N",
            shape="source:N",
            color= "data_attrib:N",#alt.condition(interval,"data_attrib:N",alt.value('lightgray')),
            tooltip=["source", "data_stage", "data_type", "data_attrib", 'count', 'size'],
        )
        #.add_selection(interval)
        .add_params( source_selection ) #,  data_type_selection)
        .transform_filter(
            # Combine both selections using logical AND
            source_selection #& data_type_selection
        )#.properties(selection=interval)
    ).interactive()
    # st.markdown("""<style> 
    #             .vega-bind {
    #             text-align:right;
    #             }</style> """, unsafe_allow_html=True)
    st.altair_chart(chart) #| chart.encode(x="size:Q"))

def disc_usage_1(tm, um, fm, w):
    #v = w["width"]
    mem = pd.DataFrame({"disc": ["Total", "Used", "Free"], "size": [tm, um, fm]})
    b = (
        alt.Chart(mem)
        .mark_bar()
        .encode(
            y=alt.Y("size:Q", axis=alt.Axis(grid=True, gridColor="grey")),
            x=alt.X("disc:N", axis=alt.Axis(grid=True, gridColor="grey")),
            color="disc:N",
            #size="size:Q",
            tooltip=['disc', 'size']
        ).properties(
            padding={"bottom": 50}
        ).configure_view(
            strokeWidth=1
        )
    )

    st.altair_chart(b, use_container_width=True)

def disc_usage(tm, um, fm, w):
    v = w['width']
    mem = pd.DataFrame({"disc": ["Total", "Used", "Free"], "size": [tm, um, fm]})

    # This formats the value as an integer for cleaner presentation in the legend/tooltip
    mem["legend_label"] = (mem["disc"] + ":" + mem["size"].astype(str) + "GB")

    # Encode theta by the value, and color by the new combined label
    base = alt.Chart(mem).encode(
        theta=alt.Theta("size:Q", stack=False)
    )
   # 4. Create the pie (arc) layer
    pie = base.mark_arc(innerRadius=int(.1*v), outerRadius=int(.3*v)).encode(
        color=alt.Color("legend_label:N", legend=None ),#alt.Legend(title="disc usage")),
        # Add tooltip for better interactivity, using the combined label field
        tooltip=["disc:N", "size:Q", alt.Tooltip("legend_label:N", title='disc usage')]
    )

    st.altair_chart(pie, use_container_width=True)


def display_storage_metrics(tm, um, fm, dfi, dff):
    c1, c2, c3 = st.columns([1.0, 1.0, 1.0])
    with c1:
        width = st_dimensions(key="c1_width")
        st.markdown("""###### <span style='color:#2d4202'><u>disc usage</u></span>""",unsafe_allow_html=True)
        disc_usage_1(tm, um, fm, width)
    with c2:
        st.markdown("""###### <span style='color:#2d4202'><u>input data usage</u></span>""",unsafe_allow_html=True)
        #ss.acquire_overview_data(dfi.values.tolist())
        filter_selection(dfi)
    with c3:
        st.markdown("""###### <span style='color:#2d4202'><u>final data usage</u></span>""",unsafe_allow_html=True)
        #ss.acquire_overview_data(dff.values.tolist())
        filter_selection(dff)


def display_folder_details(dfi, dfv, dfd, dfa, dfn):
    dfi['type'] ='image'
    dfv["type"] = "video"
    dfd['type'] ='document'
    dfa["type"] = "audio"
    dfn["type"] = "other"
    dff = pd.concat([dfi, dfv, dfd, dfa, dfn])

    # c1, c2 = st.columns([1, 1])
    # with c1:
    dff = dff.reset_index(names="file_type")
    #print(f'^^^{dff}')
    ch_count = (
        alt.Chart(dff)
        .mark_bar()
        .encode(
            x=alt.Y("count:Q"),
            y=alt.X("type:N"),
            xOffset="type:N",
            color=alt.Color("file_type:N"),
            tooltip=["type:N", "count:Q"],
        )
        .properties(
            title=f"Count- Image:{int(dfi['count'].sum())}||video:{int(dfv['count'].sum())}||document:{int(dfd['count'].sum())}||audio:{int(dfa['count'].sum())}||other:{int(dfn['count'].sum())}"
        )
    )

    ch_size = (
        alt.Chart(dff)
        .mark_bar()
        .encode(
            x=alt.X("size:Q"),
            y=alt.Y("type:N"),
            xOffset="type:N",
            color=alt.Color("file_type:N"),
            tooltip=["type:N", "size:Q"],
        )
        .properties(
            title=f"Size- Image:{int(dfi['size'].sum())}||video:{int(dfv['size'].sum())}||document:{int(dfd['size'].sum())}||audio:{int(dfa['size'].sum())}||other:{int(dfn['size'].sum())}"
        )
    )
    st.altair_chart(ch_count | ch_size)
    # with c2:
    #     dfv = dfv.reset_index()
    #     ch_count = (
    #         alt.Chart(dfv)
    #         .mark_bar()
    #         .encode(
    #             x=alt.X("count:Q", title="Videos Count"),
    #             y=alt.Y("index:N", title="Video Type", sort="y"),
    #             color=alt.Color("index:N"),
    #             tooltip=["index", "count"],
    #         )
    #         .properties(title=f"Videos Count: {int(dfv['count'].sum())}")
    #     )

    #     ch_size = (
    #         alt.Chart(dfv)
    #         .mark_bar()
    #         .encode(
    #             x=alt.X("size:Q", title="Videos Size"),
    #             y=alt.Y("index:N", title="Video Type", sort="y"),
    #             color=alt.Color("index:N"),
    #             tooltip=["index", "size"],
    #         )
    #         .properties(title=f"Videos Size: {int(dfv['size'].sum())} GB")
    #     )
    #     st.altair_chart(ch_count & ch_size)

    # with c3:
    #     dfd = dfd.reset_index()
    #     ch_count = (
    #         alt.Chart(dfd)
    #         .mark_bar()
    #         .encode(
    #             x=alt.X("count:Q", title="Documents Count"),
    #             y=alt.Y("index:N", title="Document Type", sort="y"),
    #             color=alt.Color("index:N"),
    #             tooltip=["index", "count"],
    #         )
    #         .properties(title=f"Documents Count: {int(dfd['count'].sum())}")
    #     )

    #     ch_size = (
    #         alt.Chart(dfd)
    #         .mark_bar()
    #         .encode(
    #             x=alt.X("size:Q", title="Documents Size"),
    #             y=alt.Y("index:N", title="Document Type", sort="y"),
    #             color=alt.Color("index:N"),
    #             tooltip=["index", "size"],
    #         )
    #         .properties(title=f"Documents Size .: {int(dfd['size'].sum())} GB")
    #     )
    #     st.altair_chart(ch_count & ch_size)
    # with c4:
    #     dfa = dfa.reset_index()
    #     ch_count = (
    #         alt.Chart(dfa)
    #         .mark_bar()
    #         .encode(
    #             x=alt.X("count:Q", title="Audios Count"),
    #             y=alt.Y("index:N", title="Audio Type", sort="y"),
    #             color=alt.Color("index:N"),
    #             tooltip=["index", "count"],
    #         )
    #         .properties(title=f"Audios Count: {int(dfa['count'].sum())}")
    #     )

    #     ch_size = (
    #         alt.Chart(dfa)
    #         .mark_bar()
    #         .encode(
    #             x=alt.X("size:Q", title="Audios Size"),
    #             y=alt.Y("index:N", title="Audio Type", sort="y"),
    #             color=alt.Color("index:N"),
    #             tooltip=["index", "size"],
    #         )
    #         .properties(title=f"Audio Size: {int(dfa['size'].sum())} GB")
    #     )
    #     st.altair_chart(ch_count & ch_size)

    # with c5:
    #     dfn = dfn.reset_index()
    #     ch_count = (
    #         alt.Chart(dfn)
    #         .mark_bar()
    #         .encode(
    #             x=alt.X("count:Q", title="Other File Count"),
    #             y=alt.Y("index:N", title="Other File Type", sort="y"),
    #             color=alt.Color("index:N"),
    #             tooltip=["index", "count"],
    #         )
    #         .properties(title=f"Other File Count: {int(dfn['count'].sum())}")
    #     )

    #     ch_size = (
    #         alt.Chart(dfn)
    #         .mark_bar()
    #         .encode(
    #             x=alt.X("size:Q", title="Other File Size"),
    #             y=alt.Y("index:N", title="Other File Type", sort="y"),
    #             color=alt.Color("index:N"),
    #             tooltip=["index", "size"],
    #         )
    #         .properties(title=f"Other File Size .: {int(dfn['size'].sum())} GB")
    #     )
    #     st.altair_chart(ch_count & ch_size)

def execute():
    (rdp, idp, adp, fdp, opl) = extract_folder_paths() 

    #print(f'+++{opl}')

    dfi, dff = ss.acquire_overview_data(opl)

    #print(f'---dfi and dff--> {dfi}, {dff}')
    
    #print(f'--> {rdp}')
    c1, c2 = st.columns([.1,.9])
   
    with c1:
        efs = mu.extract_user_raw_data_folders(rdp)
        #st.markdown('##### :blue[**DATA SOURCES**]')
        st.markdown("""###### <span style='color:#2d4202'><u>sources</u></span>""",unsafe_allow_html=True)
        #with st.container(height=100, border=False):        
        #st.markdown('<div class="scrollable-div">', unsafe_allow_html=True)
        with st.container(height=100, border=False):
            for ds in efs:
                st.markdown(f':green[{ds}]')
        #st.markdown('</div>', unsafe_allow_html=True)   
        #st.text_area(label="Data Sources", value=efs)
    with c2:
       display_storage_metrics(*ss.extract_server_stats(), dfi, dff)

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