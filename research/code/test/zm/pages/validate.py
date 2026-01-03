import os
import time
import streamlit as st
import altair as alt
#from stqdm import stqdm
import multiprocessing as mp
import pandas as pd
from utils.config_util import config
from utils.util import storage_stat as ss
from utils.missing_util import missing_metadata as mm
from utils.quality_util import image_quality as iq
#from utils.dedup_util.md5 import dedup_imgs as di
from utils.face_detection_util import face_trainer
from utils.dedup_util.phash import KDduplicates as kdd
from utils.dataload_util import dataload as dl
from utils.util import statusmsg_util as sm

colors = ["#6d765b", "#A5BFA6"]#["#847577", "#cfd2cd"] #["#FF6961", "#7AD7F0"]
filter_selection = []
sm.init()

def color_selectbox(n_element: int, color: str):
    js = f"""
    <script>
    // Find all the selectboxes
    var selectboxes = window.parent.document.getElementsByClassName("stSelectbox");
    
    // Select one of them
    var selectbox = selectboxes[{n_element}];
    
    // Select only the selection div
    var selectregion = selectbox.querySelector('[data-baseweb="select"]');
    
    // Modify the color
    selectregion.style.backgroundColor = '{color}';
    </script>
    """
    st.components.v1.html(js, height=0)

# get immediate child folders
def extract_user_raw_data_folders(pth):
    return next(os.walk(pth))[1]

def exe():
    (
        raw_data_path,
        duplicate_data_path,
        quality_data_path,
        missing_metadata_path,
        missing_metadata_file,
        missing_metadata_filter_file,
        metadata_file_path,
        static_metadata_file_path,
        vectordb_path,
    ) = config.data_validation_config_load()

    st.sidebar.subheader("SELECT DATA SOURCE", divider="gray")

    user_source_selected = st.sidebar.selectbox("data source folder", options=extract_user_raw_data_folders(raw_data_path),label_visibility="collapsed", index=0)
    color_selectbox(0, color=colors[0])
    # cn = mp.cpu_count()
    # npar = cn // 2

    (dfi, dfv, dfd, dfa, dfn) = ss.extract_all_folder_stats(os.path.join(raw_data_path, user_source_selected))
    dfi['type'] ='image'
    dfv["type"] = "video"
    dfd['type'] ='document'
    dfa["type"] = "audio"
    dfn["type"] = "other"
    dff = pd.concat([dfi, dfv, dfd, dfa, dfn])
    dff = dff.reset_index(names="file_type")
    st.subheader('Data Load', divider='gray')
    ca, cb, cc, cd = st.columns([.3, .2, .2, .2], gap="small")
    with ca:
        st.caption("**Media Files**")
        ch_count = (
            alt.Chart(dff)
            .mark_bar()
            .encode(
                x=alt.Y(
                    "count:Q",
                    axis=alt.Axis(grid=True, gridColor="grey"),
                    title="#files/type",
                ),
                y=alt.X(
                    "type:O",
                    axis=alt.Axis(
                        grid=True, gridColor="grey", labelAngle=-45, title=None
                    ),
                ),
                xOffset="type:O",
                color=alt.Color("file_type:N", scale=alt.Scale(scheme="dark2")),
                tooltip=["type:O", "file_type:N", "count:Q"],
                # text="Number of files by type"
            )
            # .properties(
            #     width="container",
            #     height="container",
            #     #     title=f"Count- Imgage:{int(dfi['count'].sum())}  Video:{int(dfv['count'].sum())}  Document:{int(dfd['count'].sum())}  Audio:{int(dfa['count'].sum())}  Other:{int(dfn['count'].sum())}",
            # )
        )

        ch_size = (
            alt.Chart(dff)
            .mark_bar()
            .encode(
                x=alt.Y(
                    "size:Q",
                    axis=alt.Axis(grid=True, gridColor="grey"),
                    title="size/type",
                ),
                y=alt.X("type:O", axis=alt.Axis(grid=True, gridColor="grey", labelAngle=-45),title=None),
                xOffset="type:O",
                color=alt.Color("file_type:N", scale=alt.Scale(scheme="dark2")),
                tooltip=["type:O", "file_type:N", "size:Q"],
            )
            # .properties(
            #     width="container",
            #     height="container",
            #     #     title=f"Size- Image:{int(dfi['size'].sum())} GB  Video:{int(dfv['size'].sum())} GB  Document:{int(dfd['size'].sum())} GB  Audio:{int(dfa['size'].sum())} GB  Other:{int(dfn['size'].sum())} GB"
            # )
        )
        combined_chart = alt.vconcat(ch_count, ch_size, spacing=20)
        st.altair_chart(combined_chart, use_container_width=True)

    with cb:
            (dfi, dfv, dfd, dfa, dfn) = ss.extract_all_folder_stats(os.path.join(duplicate_data_path, user_source_selected))
            dfi = dfi.reset_index(names="file_type")
            st.caption("**Duplicate Images**")
            ch = (
                alt.Chart(dfi).mark_bar()
                .encode(
                    x=alt.X(
                        "size:Q",
                        axis=alt.Axis(grid=True, gridColor="grey"),
                        # scale=alt.Scale(type="log"),
                    ),
                    y=alt.Y(
                        "count:Q",
                        axis=alt.Axis(grid=True, gridColor="grey"),
                        #scale=alt.Scale(type="log"),
                    ),
                    # size="file_type:N",
                    # shape="source:N",
                    color=alt.Color(
                        "file_type:N", scale=alt.Scale(scheme="dark2")
                    ),  # alt.condition(interval,"data_attrib:N",alt.value('lightgray')),
                    tooltip=["file_type", "count", "size"],
                )
            )
            # st.bar_chart(
            #     dfi,
            #     horizontal=False,
            #     stack=True,
            #     use_container_width=True,
            #     color=colors,
            # )

            st.altair_chart(ch, use_container_width=True)
    with cc:
            (dfi, dfv, dfd, dfa, dfn) = ss.extract_all_folder_stats(os.path.join(quality_data_path, user_source_selected))
            dfi = dfi.reset_index(names="file_type")
            st.caption("**Inferior Images**")
            ch = (
                alt.Chart(dfi).mark_bar()
                .encode(
                    x=alt.X(
                        "size:Q",
                        axis=alt.Axis(grid=True, gridColor="grey"),
                        # scale=alt.Scale(type="log"),
                    ),
                    y=alt.Y(
                        "count:Q",
                        axis=alt.Axis(grid=True, gridColor="grey"),
                        #scale=alt.Scale(type="log"),
                    ),
                    # size="file_type:N",
                    # shape="source:N",
                    color=alt.Color(
                        "file_type:N", scale=alt.Scale(scheme="dark2")
                    ),  # alt.condition(interval,"data_attrib:N",alt.value('lightgray')),
                    tooltip=["file_type", "count", "size"],
                )
            )
            # st.bar_chart(
            #     dfi,
            #     horizontal=False,
            #     stack=True,
            #     use_container_width=True,
            #     color=colors,
            # )

            st.altair_chart(ch, use_container_width=True)
            # st.bar_chart(
            #     dfi,
            #     horizontal=False,
            #     stack=True,
            #     use_container_width=True,
            #     color=colors,
            # )
   
            options = ['people','scenic','document']   # todo
            filter_selection = st.multiselect(
                '**Exclude Types**',
                options=options,
                default=['document']
            )
        
    with cd:
            #st.markdown('<div class="single-border">', unsafe_allow_html=True)
            st.caption("**Missing Metadata**")
            if os.path.exists(os.path.join( missing_metadata_path,  user_source_selected, missing_metadata_file)):       
               dict = ss.extract_stats_of_metadata_file(os.path.join( missing_metadata_path,  user_source_selected, missing_metadata_file))
               print(dict)
               df = pd.DataFrame.from_dict(dict)
               print(df)
            
               ch = alt.Chart(df).mark_bar().encode(
                    x=alt.Y(
                        "categories:N",
                        axis=alt.Axis(grid=True, gridColor="grey", title=None),
                        # scale=alt.Scale(type="log"),
                    ),
                    y=alt.X(
                        "values:Q",
                        axis=alt.Axis(grid=True, gridColor="grey", title=None),
                        #scale=alt.Scale(type="log"),
                    ),
                    # size="file_type:N",
                    # shape="source:N",
                    color=alt.Color(
                        "categories:N", scale=alt.Scale(scheme="dark2")
                    ),  # alt.condition(interval,"data_attrib:N",alt.value('lightgray')),
                    tooltip=["categories", "values"],
                )
            
            # st.bar_chart(
            #     dfi,
            #     horizontal=False,
            #     stack=True,
            #     use_container_width=True,
            #     color=colors,
            # )

               st.altair_chart(ch, use_container_width=True)
            #    st.bar_chart(
            #         df,
            #         horizontal=False,
            #         stack=True,
            #         y_label="number of images",
            #         use_container_width=True,
            #         color=alt.Color("categories:N", scale=alt.Scale(scheme="dark2")), #['#ae5a41']#,'#1b85b8','#559e83']#,'#c3cb71']#['#c09b95','#bac095','#95bac0','#9b95c0']#["#BAC095", "#A2AA70", "#848C53", "#636B2F"], #colors = ["#636B2F", "#BAC095"]
            #    )
            else:
                st.error(f"s| missing metadata file not present {missing_metadata_file}.")     
            #st.markdown('</div>', unsafe_allow_html=True)
    st.divider()    
    
    ###

    c1, c2, c3, c4 = st.columns([.3, .2, .2, .2], gap="small")
    with c1:
        # c1c = c1.container(border=False)
        # with c1c:
            if st.button("Validation Check", use_container_width=True, type='primary'):
                with st.status('validate', expanded=True) as sc1c:
                    st.write('validation check starting...')
                    results = dl.execute(user_source_selected) #test_validate_sqdm(npar)
                    if results == "success":
                        sc1c.update(label='validation complete', state='complete')
                    else:
                        sc1c.update(label='validation failed', state='error')

                    sm.show_all_msgs_by_type("validate")     

    with c2:
        # c2c= c2.container(border=False)
        # with c2c:
            if  st.button("Duplicate Check", use_container_width=True, type='primary'):
                with st.status('duplicate', expanded=True) as sc2c:
                    st.write('duplicate check starting...')
                    results = kdd.execute(user_source_selected)#test_duplicate_sqdm(npar)
                    if results == "success":
                        sc2c.update(label='duplicate complete', state='complete')
                    else:
                        sc2c.update(label='duplicate failed', state='error')  
                    sm.show_all_msgs_by_type("duplicate")

    with c3:
        # c3c = c3.container(border=False)
        # with c3c:           
            if st.button("Quality Check", use_container_width=True, type="primary"):
                with st.status(label="quality", expanded=True) as sc3c:
                    st.write("quality check starting...")
                    results = iq.execute(user_source_selected, filter_selection)
                    if results == 'success':
                        sc3c.update(label="quality complete", state="complete")
                    else:
                        sc3c.update(label="quality failed", state="error")    
                    sm.show_all_msgs_by_type('quality')


    with c4:
        # c4c = c4.container(border=False)
        # with c4c:
            if st.button("Metadata Check", use_container_width=True, type="primary"):
                with st.status(label='metadata', expanded=True) as sc4c:
                    st.write("metadata check starting...")
                    results = mm.execute(user_source_selected)  # test_metadata_sqdm(npar)
                    if results == "success":
                        sc4c.update(label="metadata complete", state="complete")
                    else:
                        sc4c.update(label="metadata failed", state="error")  
                    sm.show_all_msgs_by_type('metadata')



exe()
# if __name__=='__main__':
#     cn = mp.cpu_count()
#     execute(cn // 2)