import os
import time
import streamlit as st
import pandas as pd
import numpy as np
import streamlit.components.v1 as components
from utils.missing_util import missing_metadata as mm

mystate = st.session_state
if "btn_prsd_status" not in mystate:
    mystate.btn_prsd_status = [0] * 4

btn_labels = [
    "DATA LOAD CHECK",
    "DUP DATA CHECK",
    "DATA QUALITY CHECK",
    "METADATA CHECK",
]
unpressed_color = "#707070"
success_color = "#4CAF50"
failure_color = "#FF7F50"
wip_color = "#FFD700"


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


def btn_pressed_callback(i):
    print(i, mystate, mystate.btn_prsd_status[i - 1])
    if mystate.btn_prsd_status[i - 1] == 1 or i == 0:
        r = exec_task(i)
        mystate.btn_prsd_status[i] = r


def exec_task(iTask):  
    match iTask:
        case 0:
            print("start")
            time.sleep(10)
            print("done")
            return 1
        case 1:
            time.sleep(1)
            return 1
        case 2:
            time.sleep(1)
            return 1
        case 3:
            mm.execute()
            return 1
        case _:
            return -1
        
# get immediate chield folders
def extract_user_raw_data_folders(pth):
    return next(os.walk(pth))[1]

with st.container():
    st.header("DATA: ADD/ VALIDATE", divider="gray")
    st.sidebar.subheader("SELECT DATA SOURCE", divider="gray")

    # todo???
    st.sidebar.selectbox("data source folder", options=extract_user_raw_data_folders('/home/madhekar/work/home-media-app/data/raw-data'),label_visibility="collapsed")

    st.subheader("EXECUTE TASKS", divider="gray")
    c0, c1, c2, c3 = st.columns((1, 1, 1, 1), gap="small")
    with c0:
        st.button(
            btn_labels[0],
            key="g0",
            on_click=btn_pressed_callback,
            args=(0,),
            use_container_width=True,
        )
        st.divider()
        chart_data = pd.DataFrame(
            abs(np.random.randn(1, 4)) * 100,
            columns=["images", "text", "video", "audio"],
        )
        st.bar_chart(
            chart_data,
            horizontal=False,
            stack=False,
            y_label="number of files",
            use_container_width=True,
        )
        st.divider()
        st.text_area("data load msgs:")
    with c1:
        st.button(
            btn_labels[1], key="g1", on_click=btn_pressed_callback, args=(1,), use_container_width=True
        )
        st.divider()
        chart_data = pd.DataFrame(
            abs(np.random.randn(1, 4)) * 100,
            columns=["images", "text", "video", "audio"],
        )
        st.bar_chart(
            chart_data,
            horizontal=False,
            stack=False,
            y_label="number of files",
            use_container_width=True,
        )
        st.divider()
        st.text_area("duplicate data msgs:")
    with c2:
        st.button(
            btn_labels[2], key="g2", on_click=btn_pressed_callback, args=(2,), use_container_width=True
        )
        st.divider()
        chart_data = pd.DataFrame(
            abs(np.random.randn(1, 4)) * 100,
            columns=["images", "text", "video", "audio"],
        )
        st.bar_chart(
            chart_data,
            horizontal=False,
            stack=False,
            y_label="number of files",
            use_container_width=True,
        )
        st.divider()
        st.text_area("quality check msgs:")
    with c3:
        st.button(btn_labels[3], key="g3", on_click=btn_pressed_callback, args=(3,), use_container_width=True)
        st.divider()
        chart_data = pd.DataFrame(
            abs(np.random.randn(1, 4)) * 100,
            columns=["images", "text", "video", "audio"],
        )
        st.bar_chart(
            chart_data,
            horizontal=False,
            stack=False,
            y_label="number of files",
            use_container_width=True,
        )
        st.divider()
        st.text_area("METADATA CHECK msgs:")
    ChkBtnStatusAndAssigncolor()
