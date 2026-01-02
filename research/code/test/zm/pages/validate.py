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
    ca, cb, cc, cd = st.columns([.4, .2, .2, .2], gap="small")
    with ca:
        ch_count = (
            alt.Chart(dff)
            .mark_bar()
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
                 width="container",
                 height="container"
            #     title=f"Count- Imgage:{int(dfi['count'].sum())}  Video:{int(dfv['count'].sum())}  Document:{int(dfd['count'].sum())}  Audio:{int(dfa['count'].sum())}  Other:{int(dfn['count'].sum())}",
             )
        )

        ch_size = (
            alt.Chart(dff)
            .mark_bar()
            .encode(
                x=alt.X(
                    "size:Q",
                    axis=alt.Axis(grid=True, gridColor="grey"),
                    title="Storage size by file type",
                ),
                y=alt.Y("type:N", axis=alt.Axis(grid=True, gridColor="grey")),
                xOffset="type:N",
                color=alt.Color("file_type:N", scale=alt.Scale( scheme='dark2')),
                tooltip=["type:N", "file_type:N", "size:Q"],
            )
            # .properties(
            #     title=f"Size- Image:{int(dfi['size'].sum())} GB  Video:{int(dfv['size'].sum())} GB  Document:{int(dfd['size'].sum())} GB  Audio:{int(dfa['size'].sum())} GB  Other:{int(dfn['size'].sum())} GB"
            # )
        )
        st.altair_chart(ch_count & ch_size, use_container_width=True)


    # with ca:
    #     cac= ca.container(border=False)
    #     with cac:
    #         c01, c02, c03, c04 = st.columns([1,1,1,1], gap="small")
    #         with c01:
    #             st.caption("**Images Loaded**")
    #             st.bar_chart(
    #                 dfi,
    #                 horizontal=False,
    #                 stack=True,
    #                 #y_label="total size(MB) & count of images",
    #                 use_container_width=True,
    #                 color=colors, #, "#636B2F", "#3D4127"] # colors
    #                 )
    #         with c02:
    #             st.caption("**Videos Loaded**")
    #             st.bar_chart(
    #                 dfv,
    #                 horizontal=False,
    #                 stack=True,
    #                 # y_label="total size(MB) & count of images",
    #                 use_container_width=True,
    #                 color=colors,  # , "#636B2F", "#3D4127"] # colors
    #             )
    #         with c03:
    #             st.caption("**Documents Loaded**")
    #             st.bar_chart(
    #                 dfd,
    #                 horizontal=False,
    #                 stack=True,
    #                 # y_label="total size(MB) & count of images",
    #                 use_container_width=True,
    #                 color=colors,  # , "#636B2F", "#3D4127"] # colors
    #             )
    #         with c04:
    #             st.caption("**Others Loaded**")
    #             st.bar_chart(
    #                 dfn,
    #                 horizontal=False,
    #                 stack=True,
    #                 # y_label="total size(MB) & count of images",
    #                 use_container_width=True,
    #                 color=colors,  # colors=["#D4DE95", "#BAC095"],  # ,
    #             )

    with cb:
        cbc = cb.container(border=False)
        with cbc:
            (dfi, dfv, dfd, dfa, dfn) = ss.extract_all_folder_stats(os.path.join(duplicate_data_path, user_source_selected))
            st.caption("**Duplicate Images Archived**")
            st.bar_chart(
                dfi,
                horizontal=False,
                stack=True,
                use_container_width=True,
                color=colors,
            )


    with cc:
        ccc= cc.container(border=False)
        with ccc:
            (dfi, dfv, dfd, dfa, dfn) = ss.extract_all_folder_stats(os.path.join(quality_data_path, user_source_selected))
            st.caption("**Inferior Quality Images Found**")
            st.bar_chart(
                dfi,
                horizontal=False,
                stack=True,
                use_container_width=True,
                color=colors,
            )
   
            options = ['people','scenic','document']   # todo
            filter_selection = st.multiselect(
                '**Purge: Select Image Types**',
                options=options,
                default=['document']
            )
        

    with cd:
        cdc= cd.container(border=False)        
        with cdc:
            if os.path.exists(os.path.join( missing_metadata_path,  user_source_selected, missing_metadata_file)):       
               dict = ss.extract_stats_of_metadata_file(os.path.join( missing_metadata_path,  user_source_selected, missing_metadata_file))
               #print(dict)
               df = pd.DataFrame.from_dict(dict, orient='index',columns=['number'])
              #print(df)
            
               st.bar_chart(
                    df,
                    horizontal=False,
                    stack=True,
                    y_label="number of images",
                    use_container_width=True,
                    color= colors[0] #['#ae5a41']#,'#1b85b8','#559e83']#,'#c3cb71']#['#c09b95','#bac095','#95bac0','#9b95c0']#["#BAC095", "#A2AA70", "#848C53", "#636B2F"], #colors = ["#636B2F", "#BAC095"]
               )
            else:
                st.error(f"s| missing metadata file not present {missing_metadata_file}.")     

    st.divider()    
    
    ###

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1], gap="small")
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