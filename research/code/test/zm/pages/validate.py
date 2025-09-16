import os
import time
import streamlit as st
from stqdm import stqdm
import multiprocessing as mp
import pandas as pd
from utils.config_util import config
from utils.util import storage_stat as ss
from utils.missing_util import missing_metadata as mm
from multipage_app.utils.quality_util import image_quality as iq
from utils.dedup_util.md5 import dedup_imgs as di
from utils.dedup_util.phash import KDduplicates as kdd
from utils.dataload_util import dataload as dl
from utils.util import statusmsg_util as sm

colors = ["#FF6961", "#7AD7F0"]

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

    user_source_selected = st.sidebar.selectbox("data source folder", options=extract_user_raw_data_folders(raw_data_path),label_visibility="collapsed", index=1)
    color_selectbox(0, "#FF6961")
    # cn = mp.cpu_count()
    # npar = cn // 2

    (dfi, dfv, dfd, dfa, dfn) = ss.extract_all_folder_stats(os.path.join(raw_data_path, user_source_selected))

    st.subheader('Data Load', divider='gray')
    ca, cb, cc, cd = st.columns([1, 1, 1, 1], gap="small")

    with ca:
        cac= ca.container(border=False)
        with cac:
            c01, c02, c03, c04 = st.columns([1,1,1,1], gap="small")
            with c01:
                st.caption("**Images Loaded**")
                st.bar_chart(
                    dfi,
                    horizontal=False,
                    stack=True,
                    #y_label="total size(MB) & count of images",
                    use_container_width=True,
                    color=colors, #, "#636B2F", "#3D4127"] # colors
                    )
            with c02:
                st.caption("**Videos Loaded**")
                st.bar_chart(
                    dfv,
                    horizontal=False,
                    stack=True,
                    # y_label="total size(MB) & count of images",
                    use_container_width=True,
                    color=colors,  # , "#636B2F", "#3D4127"] # colors
                )
            with c03:
                st.caption("**Documents Loaded**")
                st.bar_chart(
                    dfd,
                    horizontal=False,
                    stack=True,
                    # y_label="total size(MB) & count of images",
                    use_container_width=True,
                    color=colors,  # , "#636B2F", "#3D4127"] # colors
                )
            with c04:
                st.caption("**Others Loaded**")
                st.bar_chart(
                    dfn,
                    horizontal=False,
                    stack=True,
                    # y_label="total size(MB) & count of images",
                    use_container_width=True,
                    color=colors,  # colors=["#D4DE95", "#BAC095"],  # ,
                )

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
            st.caption("**Bad Quality Images Archived**")
            st.bar_chart(
                dfi,
                horizontal=False,
                stack=True,
                use_container_width=True,
                color=colors,
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
                    st.write('validation check start')
                    results = dl.execute(user_source_selected) #test_validate_sqdm(npar)
                    if results == "success":
                        sc1c.update(label='validation complete...', state='complete')
                    else:
                        sc1c.update(label='validation failed...', state='error')   

                    msgs = sm.get_message_by_type("load")  
                    if msgs:
                        for k,v in msgs.items():
                            if k == 's':
                              st.info(str(v))
                            elif k == 'w':
                              st.warning(str(v))   
                            else:
                              st.error(str(v))      

    with c2:
        # c2c= c2.container(border=False)
        # with c2c:
            if  st.button("Duplicate Check", use_container_width=True, type='primary'):
                with st.status('duplicate', expanded=True) as sc2c:
                    st.write('duplicate image check start')
                    results = kdd.execute(user_source_selected)#test_duplicate_sqdm(npar)
                    if results == "success":
                        sc2c.update(label='duplicate complete...', state='complete')
                    else:
                        sc2c.update(label='duplicate failed...', state='error')  
                    msgs = sm.get_message_by_type("duplicate")
                    if msgs:
                      for k, v in msgs.items():
                        if k == "s":
                            st.info(str(v))
                        elif k == "w":
                            st.warning(str(v))
                        else:
                            st.error(str(v))

    with c3:
        # c3c = c3.container(border=False)
        # with c3c:            
            if st.button("Quality Check", use_container_width=True, type="primary"):
                with st.status(label="quality", expanded=True) as sc3c:
                    st.write("quality check start")
                    results = iq.execute(user_source_selected)  # test_quality_sqdm(npar)
                    print('---quality-->', results)
                    if results == 'success':
                        sc3c.update(label="quality complete", state="complete")
                    else:
                        sc3c.update(label="quality failed", state="error")    

                    msgs = sm.get_message_by_type("quality")
                    if msgs:
                      for k, v in msgs.items():
                        if k == "s":
                            st.info(str(v))
                        elif k == "w":
                            st.warning(str(v))
                        else:
                            st.error(str(v))

    with c4:
        # c4c = c4.container(border=False)
        # with c4c:
            if st.button("Metadata Check", use_container_width=True, type="primary"):
                with st.status(label='quality', expanded=True) as sc4c:
                    st.write("metadata check start")
                    results = mm.execute(user_source_selected)  # test_metadata_sqdm(npar)
                    if results == "success":
                        sc4c.update(label="metadata complete", state="complete")
                    else:
                        sc4c.update(label="metadata failed", state="error")  
                    msgs = sm.get_message_by_type("metadata")
                    if msgs:
                      for k, v in msgs.items():
                        if k == "s":
                            st.info(str(v))
                        elif k == "w":
                            st.warning(str(v))
                        else:
                            st.error(str(v))


exe()
# if __name__=='__main__':
#     cn = mp.cpu_count()
#     execute(cn // 2)