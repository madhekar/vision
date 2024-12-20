import os
import time
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import streamlit.components.v1 as components
from utils.config_util import config
from utils.missing_util import missing_metadata as mm
from utils.quality_util import image_quality as iq
from utils.dedup_util import dedup_imgs as di
from utils.dataload_util import dataload as dl
from utils.util import storage_stat as ss

mystate = st.session_state
if "btn_prsd_status" not in mystate:
    mystate.btn_prsd_status = [0] * 4

if "msgs" not in mystate:
    mystate.msgs = {"load": [], "duplicate": [], "quality": [], "metadata": []}

def add_messages(msg_type, message):
    print(msg_type, message)
    mystate.msgs[msg_type].append(message) 

    print(mystate.msgs[msg_type])   

def get_message_by_type(tmsg):
    return mystate.msgs[tmsg] #[msg for msg_type, msg  in mystate.msgs.items() if msg_type == tmsg]    

btn_labels = [
    "DATA LOAD VALIDATE",
    "IMG: PURGE DUPLICATES",
    "IMG: PURGE BAD QUALITY",
    "IMG: METADATA VALIDATE",
]
unpressed_color = "#5a5255"#"#636B2F"  # colors = ["#BAC095", "#636B2F"]
success_color = '#559e83' #"#BAC095"
failure_color = '#ae5a41' #"#6B2F45"
wip_color = "#1b85b8"#"#998E1A"

colors = ["#ae5a41", "#1b85b8"]#["#636B2F", "#BAC095"]

def ChangeButtoncolor(widget_label, prsd_status):
    btn_bg_color = success_color if prsd_status == True else unpressed_color
    if prsd_status == 0:
        btn_bg_color = unpressed_color
    elif prsd_status == 1:
        btn_bg_color = success_color
    elif prsd_status == 2:
        btn_bg_color = failure_color
    else:
        btn_bg_color = wip_color

    htmlstr = f"""
        <script>
            var elements = window.parent.document.querySelectorAll('button');
            for (var i = 0; i < elements.length; ++i) {{ 
                if (elements[i].innerText == '{widget_label}') {{ 
                    elements[i].style.background = '{btn_bg_color}'
                }}
            }}
        </script>
        """
    components.html(f"{htmlstr}", height=0, width=0)


def ChkBtnStatusAndAssigncolor():
    for i in range(len(btn_labels)):
        ChangeButtoncolor(btn_labels[i], mystate.btn_prsd_status[i])


def btn_pressed_callback(i, user_source):
    print(i, mystate, mystate.btn_prsd_status[i - 1])
    if mystate.btn_prsd_status[i - 1] == 1 or i == 0:
        r = exec_task(i, user_source)
        mystate.btn_prsd_status[i] = r

        
# get immediate chield folders
def extract_user_raw_data_folders(pth):
    return next(os.walk(pth))[1]

def execute():        

    # extract config settings  
    (
        raw_data_path,
        duplicate_data_path,
        quality_data_path,
        missing_metadata_path,
        missing_metadata_file,
        metadata_file_path,
        static_metadata_file_path,
        vectordb_path
    ) = config.data_validation_config_load()

    # get source source folder
    user_source_selected = st.sidebar.selectbox("data source folder", options=extract_user_raw_data_folders(raw_data_path),label_visibility="collapsed")

    with st.container():
        st.header("DATA: VALIDATE", divider="gray")
        st.sidebar.subheader("SELECT DATA SOURCE", divider="gray")

        st.subheader("VALIDATION TASKS", divider="gray")
        c0, c1, c2, c3 = st.columns((1, 1, 1, 1), gap="small")
        with c0:
            st.button(
                btn_labels[0],
                key="g0",
                on_click=btn_pressed_callback,
                args=(0,user_source_selected),
                use_container_width=True,
            )
            st.divider()

            (dfi, dfv, dfd, dfa, dfn) = ss.extract_all_folder_stats(os.path.join(raw_data_path, user_source_selected))

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
            st.divider()
            status_con = st.status("**load data task msgs...**", expanded=True, state="running")
            with status_con:
                msgs = get_message_by_type("load")  
                print('--->', len(msgs))
                if msgs:
                    for m in msgs:
                        st.info(str(m))
    
        with c1:
            st.button(
                btn_labels[1], key="g1", on_click=btn_pressed_callback, args=(1,user_source_selected), use_container_width=True
            )
            st.divider()
            (dfi, dfv, dfd, dfa, dfn) = ss.extract_all_folder_stats(duplicate_data_path)
            st.caption('**Duplicate Images Archived**')
            st.bar_chart(
                dfi,
                horizontal=False,
                stack=True,
                use_container_width=True,
                color=colors,
            )
            st.divider()
            status_con = st.status('**de-duplicate data task msgs...**', expanded=True)
            with status_con:
                msgs = get_message_by_type("duplicate")
                if msgs:
                    for m in msgs:
                        print(m)
                        st.write(str(m))
        with c2:
            st.button(
                btn_labels[2], key="g2", on_click=btn_pressed_callback, args=(2, user_source_selected), use_container_width=True
            )
            st.divider()
            (dfi, dfv, dfd, dfa, dfn) = ss.extract_all_folder_stats(quality_data_path)
            st.caption('**Bad Quality Images Archived**')
            st.bar_chart(
                dfi,
                horizontal=False,
                stack=True,
                use_container_width=True,
                color=colors,
            )
            st.divider()
            status_con = st.status("**data quality check task msgs...**", expanded=True, state="running")
            with status_con:
                msgs = get_message_by_type("quality")
                if msgs:
                    for m in msgs:
                        print(m)
                        st.write(str(m))
        with c3:
            st.button(btn_labels[3], key="g3", on_click=btn_pressed_callback, args=(3,user_source_selected), use_container_width=True)
            st.divider()
            st.caption("**Images With Missing Metadata**")
                   
            dict = ss.extract_stats_of_metadata_file(os.path.join( missing_metadata_path,  missing_metadata_file))
            print(dict)
            df = pd.DataFrame.from_dict([dict])
            st.bar_chart(
                df,
                horizontal=False,
                stack=False,
                y_label="number of files",
                use_container_width=True,
                color= ['#ae5a41','#1b85b8','#559e83','#c3cb71']#['#c09b95','#bac095','#95bac0','#9b95c0']#["#BAC095", "#A2AA70", "#848C53", "#636B2F"], #colors = ["#636B2F", "#BAC095"]
            )
            # fig = px.pie(df, names=df.columns, values=df.values[0],color_discrete_sequence=colors, height=355 )
            # fig.update_traces( textinfo="percent+value")
            # st.plotly_chart(fig, use_container_width=False)
            st.divider()
            status_con = st.status("**metadata check task msgs...**", expanded=True)
            with status_con:
                msgs = get_message_by_type("metadata")
                if msgs:
                    for m in msgs:
                        print(m)
                        st.write(str(m))
        ChkBtnStatusAndAssigncolor()

def exec_task(iTask, user_source):
    match iTask:
        case 0:  
            # load images check
            task_name = 'data load'
            add_messages('load', f"starting {task_name} prpcess")
            dl.execute(user_source)
            add_messages("load", f"done {task_name} prpcess")
            return 1
        case 1:  # duplicate images check
            task_name = 'de-duplicate files'
            add_messages('duplicate', f"starting {task_name} prpcess")
            di.execute(user_source)
            add_messages("duplicate", f"done {task_name} prpcess")
            return 1
        case 2:  # image sharpness/ quality check
            task_name = 'image quality check'
            add_messages("quality", f"starting {task_name} prpcess")
            iq.execute()
            add_messages('quality', f"done {task_name} prpcess")
            return 1
        case 3:  # missing metadata check
            task_name = "missing metadata"
            add_messages("metadata", f"starting {task_name} prpcess")
            mm.execute()
            add_messages('metadata',f"done {task_name} prpcess")
            return 1
        case _:
            return -1        

execute()