import os
import time
import streamlit as st
import pandas as pd
import numpy as np
import streamlit.components.v1 as components



st.set_page_config(
    page_title="zesha: Home Media Portal (HMP)",
    page_icon="/home/madhekar/work/zsource/zesha-high-resolution-logo.jpeg",
    initial_sidebar_state="auto",
    layout="wide",
)

def load_css(css_path):
    with open(file=css_path) as f:
        s = f"<style>{f.read()}</style>"
        st.html(s)


css_path = os.path.join("assets", "styles.css")

load_css(css_path)

mystate = st.session_state
if "btn_prsd_status" not in mystate:
    mystate.btn_prsd_status = [False] * 6

btn_labels = ["Data Load Check", "Duplicate Data Check", "Data Quality Check", "Metadata Check", "Metadata Correction", "Data Loader Check" ]
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
    #mystate.btn_prsd_status = [False] * 6
    if 
    r = exec_task(i)
    mystate.btn_prsd_status[i - 1] = r


def exec_task(iTask):
    match iTask:
        case 1:
            print('start')
            time.sleep(10)
            print('done')
            return 1
        case 2:
            time.sleep(1)
            return 1
        case 3:
            time.sleep(1)
            return 2
        case 4:
            return 0
        case 5:
            return 0
        case 6: 
            return 0
        case _:
            return -1

with st.container():
    st.title("workflow")
    st.subheader('execute tasks: ')
    st.divider()
    c1, c2, c3, c4, c5,c6 = st.columns ((1, 1, 1, 1, 1, 1), gap='small')
    with c1:
      c1.button("Data Load Check", key="g1", on_click=btn_pressed_callback, args=(1,))
      c1.divider()
      chart_data = pd.DataFrame(abs(np.random.randn(1, 4)) *100 , columns=["images", "text", "video", "audio"])
      st.bar_chart(chart_data, horizontal=False, stack=False, y_label="number of files", use_container_width=True)
      c1.divider()
      c1.text_area("data load msgs:")
    with c2:  
      c2.button("Duplicate Data Check", key="g2", on_click=btn_pressed_callback, args=(2,))
      c2.divider()
      chart_data = pd.DataFrame(abs(np.random.randn(1, 4)) *100 , columns=["images", "text", "video", "audio"])
      st.bar_chart(chart_data, horizontal=False, stack=False, y_label="number of files", use_container_width=True)
      st.divider()
      st.text_area("duplicate data msgs:")
    with c3:  
       c3.button("Data Quality Check", key="g3", on_click=btn_pressed_callback, args=(3,))
       c3.divider()
       chart_data = pd.DataFrame(abs(np.random.randn(1, 4)) *100 , columns=["images", "text", "video", "audio"])
       st.bar_chart(chart_data, horizontal=False, stack=False , y_label="number of files", use_container_width=True)
       st.divider()
       st.text_area("quality check msgs:")
    with c4:         
       c4.button("Metadata Check", key="g4", on_click=btn_pressed_callback, args=(4,))
       c4.divider()
       chart_data = pd.DataFrame(abs(np.random.randn(1, 4)) *100 , columns=["images", "text", "video", "audio"])
       st.bar_chart(chart_data, horizontal=False, stack=False, y_label="number of files", use_container_width=True)
       st.divider()
       st.text_area("metadata check msgs:")       
    with c5:         
       c5.button("Metadata Correction", key="g5", on_click=btn_pressed_callback, args=(5,))
       c5.divider()
       chart_data = pd.DataFrame(abs(np.random.randn(1, 4)) *100 , columns=["images", "text", "video", "audio"])
       st.bar_chart(chart_data, horizontal=False, stack=False, y_label="number of files", use_container_width=True)
       st.divider()
       st.text_area("fix metadata msgs:")
    with c6:     
       c6.button("Data Loader Check", key="g6", on_click=btn_pressed_callback, args=(6,))
       c6.divider()
       chart_data = pd.DataFrame(abs(np.random.randn(1, 4)) *100 , columns=["images", "text", "video", "audio"])
       st.bar_chart(chart_data, horizontal=False, stack=False, y_label="number of files", use_container_width=True)
       st.divider()
       st.text_area("vectordb load msgs:")
    ChkBtnStatusAndAssigncolor() 