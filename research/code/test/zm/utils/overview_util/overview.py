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
    #interval = alt.selection_interval(encodings=['x','y'])
    # 1. Define the first dropdown selection
    source_selection = alt.selection_point(
        fields=["source"], 
        bind="legend", 
        name="source"
    )
    # For a dropdown *widget*, use bind=alt.binding_select(options=...)
    source_dropdown = alt.binding_select(
        options=sorted(df["source"].unique().tolist()), name="Select source"
    )

    source_selection = alt.selection_point(
        name="SelectSource",
        fields=["source"], 
        bind=source_dropdown,
        value="madhekar"
        )

    #df['combined_category'] = df['data_type'] +',' + df['data_attrib']

    # # 2. Define the second dropdown selection
    # data_stage_dropdown = alt.binding_select(
    #     options=sorted(df["data_stage"].unique().tolist()), name="Select data stage "
    # )
    # data_stage_selection = alt.selection_point(
    #     fields=["data_stage"], bind=data_stage_dropdown
    # )

    #3. Define the third selection
    # data_type_dropdown = alt.binding_select(
    #     options=sorted(df["data_type"].unique().tolist()), name="Select data type "
    # )
    # data_type_selection = alt.selection_point(
    #     fields=["data_type"], bind=data_type_dropdown
    # )


    df['size'] = df['size'].astype(float)
    df['count'] = df['count'].astype(int)

    base = alt.Chart(df).encode(

    y=alt.Y('data_type:N', sort='-x', axis=alt.Axis(grid=True, gridColor="grey"), title='data Type'), # Sort descending by x-value
    yOffset="data_attrib:N",
    color=alt.Color("data_attrib:N", scale=alt.Scale( scheme='dark2')),
    tooltip=['data_type', 'data_attrib','count', 'size']
    ).transform_filter(
      (alt.datum.count > 0)
    )
    # .properties(
    # title= stage,#'File count and Size by Type',
    # )

    # Bar chart for Size (MB)
    size_chart = (
        base.mark_bar(color="skyblue", opacity=0.7)
        .encode(
            x=alt.X(
                "size:Q",
                axis=alt.Axis(grid=True, gridColor="grey"),
                title="folder Size",
            ),
        )
        .transform_filter(
            # Combine both selections using logical AND
            source_selection  # & data_type_selection
        )
        .add_params(source_selection)
    )

    # Text labels for count on the bars
    text_count = size_chart.mark_text(
        align='left',
        baseline='middle',
        dx=5 # Nudges text to the right of the bar
    ).encode(
        x='size:Q',
        text='count:Q',
        color=alt.value('black')
    ).transform_filter(
        # Combine both selections using logical AND
        source_selection #& data_type_selection
    ).add_params( source_selection )
    # .facet(
    # column='data_attrib:N'
    # )
    #.properties(selection=interval)
    #.interactive()
    # st.markdown("""<style> 
    #             .vega-bind {
    #             text-align:right;
    #             }</style> """, unsafe_allow_html=True)
    st.altair_chart(size_chart + text_count, use_container_width=True) #| chart.encode(x="size:Q"))
    # Combine the bar chart and text labels
    # chart = size_chart + text_count
    # st.altair_chart(chart, use_container_width=True)
    # Create the chart
    # chart = (
    #     alt.Chart(df)
    #     .mark_bar()
    #     .encode(
    #         x=alt.X("count:Q", axis=alt.Axis(grid=True, gridColor='grey')),
    #         y=alt.Y("size:Q", axis=alt.Axis(grid=True, gridColor="grey")),
    #         #size="count:Q",
    #         shape="data_type:N",
    #         color= alt.Color("data_attrib:N", scale=alt.Scale( scheme='dark2')),#alt.condition(interval,"data_attrib:N",alt.value('lightgray')),
    #         tooltip=["source", "data_stage", "data_type", "data_attrib", 'count', 'size'],
    #     )
    #.add_selection(interval)


def disc_usage_1(tm, um, fm, w):
    #v = w["width"]
    mem = pd.DataFrame({"disc": ["Total", "Used", "Free"], "size": [tm, um, fm]})

    bar = (
        alt.Chart(mem)
        .mark_bar(opacity=0.7)
        .encode(
            y=alt.Y("size:Q", axis=alt.Axis(grid=True, gridColor="grey"), title="Disc Storage Size in GB"),
            x=alt.X("disc:N", axis=alt.Axis(grid=True, gridColor="grey"), title="Disc Storage Category"),
            color=alt.Color("disc:N", scale=alt.Scale( scheme='dark2'), legend=alt.Legend(labelFontWeight="bold",labelColor="gray"))#range=['#5A86AD', '#A3B18A', '#C17C74']))
        )
    )
    text = bar.mark_text(
        align="center",
        baseline="middle",
        dy=-5,
        fontWeight="bold",
        color="black"
    ).encode(text="size")

    st.altair_chart(bar + text, use_container_width=True)

    # b = (
    #     alt.Chart(mem)
    #     .mark_bar()
    #     .encode(
    #         y=alt.Y("size:Q", axis=alt.Axis(grid=True, gridColor="grey")),
    #         x=alt.X("disc:N", axis=alt.Axis(grid=True, gridColor="grey")),
    #         color="disc:N",
    #         #size="size:Q",
    #         tooltip=['disc', 'size']
    #     ).properties(
    #         padding={"bottom": 50}
    #     ).configure_view(
    #         strokeWidth=1
    #     )
    # )

    # st.altair_chart(b, use_container_width=True)

def disc_usage(tm, um, fm, w):

    v = w['width'] if w is not None else 400
    mem = pd.DataFrame({"disc": ["Total", "Used", "Free"], "size": [tm, um, fm]})

    # This formats the value as an integer for cleaner presentation in the legend/tooltip
    mem["legend_label"] = (mem["disc"] + ":" + mem["size"].astype(str) + "GB")

    # Encode theta by the value, and color by the new combined label
    base = alt.Chart(mem).encode(
        theta=alt.Theta("size:Q").stack(True),
        radius=alt.Radius("size").scale(type="sqrt", zero=True),
        color=alt.Color("legend_label:N", scale=alt.Scale(scheme="dark2"))
    )

   # 4. Create the pie (arc) layer innerRadius=int(0.05 * v), outerRadius=int(0.2 * v)
    pie = base.mark_arc(opacity=0.7, innerRadius=int(0.1 * v), outerRadius=int(0.3 * v), stroke="#fff").encode(
        # alt.Theta("size:Q").stack(True),
        # alt.Radius("size").scale(type="sqrt", zero=True),
        # color=alt.Color("legend_label:N", scale=alt.Scale(scheme="dark2")),  # alt.Legend(title="disc usage")),
        # Add tooltip for better interactivity, using the combined label field
        tooltip=["disc:N", "size:Q", alt.Tooltip("legend_label:N")],
    )
    text = base.mark_text(align='center', radiusOffset=10, color="black").encode(text="size:Q")
    st.altair_chart(pie + text, use_container_width=True)


def display_storage_metrics(tm, um, fm, dfi, dff):
    c1, c2, c3 = st.columns([.3, 1.0, 1.0])
    with c1:
        #st.markdown('<p class="vertical-text">DISK usage</p>', unsafe_allow_html=True)
        width = st_dimensions(key="c1_width")
        st.markdown("""##### <span style='color:#2d4202'><u>DISK usage</u></span>""",unsafe_allow_html=True)
        disc_usage(tm, um, fm, width)
    with c2:
        #st.markdown('<p class="vertical-text">input data folder usage</p>', unsafe_allow_html=True)
        st.markdown("""##### <span style='color:#2d4202'><u>INPUT DATA FOLDER usage</u></span>""",unsafe_allow_html=True)
        #ss.acquire_overview_data(dfi.values.tolist())
        filter_selection(dfi)
    with c3:
        # st.markdown(
        #     '<p class="vertical-text">final data folder usage</p>',
        #     unsafe_allow_html=True,
        # )
        st.markdown("""##### <span style='color:#2d4202'><u>FINAL DATA FOLDER usage</u></span>""",unsafe_allow_html=True)
        #ss.acquire_overview_data(dff.values.tolist())
        filter_selection(dff)


def display_folder_details(dfi, dfv, dfd, dfa, dfn):
    dfi['type'] ='image'
    dfv["type"] = "video"
    dfd['type'] ='text'
    dfa["type"] = "audio"
    dfn["type"] = "other"
    dff = pd.concat([dfi, dfv, dfd, dfa, dfn])

    # c1, c2 = st.columns([1, 1])
    # with c1:
    dff = dff.reset_index(names="file_type")

    ch_count = (
        alt.Chart(dff)
        .mark_bar(opacity=0.7)
        .encode(
            x=alt.Y(
                "count:Q",
                axis=alt.Axis(grid=True, gridColor="grey"),
                title="Number of files by type",
            ),
            y=alt.X("type:N", axis=alt.Axis(grid=True, gridColor="grey")),
            xOffset="type:N",
            color=alt.Color("file_type:N", scale=alt.Scale(scheme="dark2")),
            tooltip=["type:N", "file_type:N", "count:Q"],
            # text="Number of files by type"
        )
        .properties(
            # width='container',
            title=f"File Count- Imgage:{int(dfi['count'].sum())}  Video:{int(dfv['count'].sum())}  Text:{int(dfd['count'].sum())}  Audio:{int(dfa['count'].sum())}  Other:{int(dfn['count'].sum())}"
        )
        .configure(padding={"left": 0, "top": 5, "right": 0, "bottom": 5})
    )

    ch_size = (
        alt.Chart(dff)
        .mark_bar(opacity=0.7)
        .encode(
            x=alt.X(
                "size:Q",
                axis=alt.Axis(grid=True, gridColor="grey"),
                title="Storage size by file type",
                #scale=alt.Scale(type="log")
            ),
            y=alt.Y("type:N", axis=alt.Axis(grid=True, gridColor="grey")),
            xOffset="type:N",
            color=alt.Color("file_type:N", scale=alt.Scale( scheme='dark2')),
            tooltip=["type:N", "file_type:N", "size:Q"],
        )
        .properties(
            #width='container',
            title=f"Files Size- Image:{int(dfi['size'].sum())} GB  Video:{int(dfv['size'].sum())} GB  Text:{int(dfd['size'].sum())} GB  Audio:{int(dfa['size'].sum())} GB  Other:{int(dfn['size'].sum())} GB"
        ).configure(
            padding={"left":0, "top":5, "right": 0, "bottom": 5}
        )
    )
    c1, c2 = st.columns([.5,.5])
    with c1:
       st.altair_chart(ch_count, use_container_width=True)
    with c2:   
       st.altair_chart(ch_size, use_container_width=True)

def execute():
    (rdp, idp, adp, fdp, opl) = extract_folder_paths() 

    dfi, dff = ss.acquire_overview_data(opl)
   
    efs = mu.extract_user_raw_data_folders(rdp)

    st.sidebar.markdown('##### :blue[**DATA SOURCES**]')
    #st.sidebar.markdown("""#### <span style='color:#2d4202'><u>SOURCES</u></span>""",unsafe_allow_html=True)
    #with st.container(height=100, border=False):        
    #st.markdown('<div class="scrollable-div">', unsafe_allow_html=True)
    with st.container(height=100, border=True):
        for ds in efs:
            st.sidebar.write(f'**{ds}**')
            #st.sidebar.caption(ds)
    st.sidebar.divider()        
    #st.markdown('</div>', unsafe_allow_html=True)   
    #st.text_area(label="Data Sources", value=efs)
    
    st.markdown(
        """
    <style>
            .st-emotion-cache-1cw0ubf
                {
                   display:none;
                }    

            # .st-emotion-cache-vi2kj0
            # {
            #         display:none;
            # }    
    </style>
    """,
        unsafe_allow_html=True,
    )
    display_storage_metrics(*ss.extract_server_stats(), dfi, dff)

    st.divider()

    st.markdown("""##### <span style='color:#2d4202'><u>**RAW DATA FOLDER**</u></span>""",unsafe_allow_html=True)
    display_folder_details(*ss.extract_all_folder_stats(rdp))
    #st.divider()

    st.markdown("""##### <span style='color:#2d4202'><u>**INPUT DATA FOLDER**</u></span>""",unsafe_allow_html=True)
    display_folder_details(*ss.extract_all_folder_stats(idp))
    #st.divider()

    st.markdown("""##### <span style='color:#2d4202'><u>**FINAL DATA FOLDER**</u></span>""",unsafe_allow_html=True)
    display_folder_details(*ss.extract_all_folder_stats(fdp))
    #st.divider()

    st.markdown("""##### <span style='color:#2d4202'><u>**APP DATA FOLDER**</u></span>""",unsafe_allow_html=True)
    display_folder_details(*ss.extract_all_folder_stats(adp))

# if __name__ == "__main__":
#     execute()