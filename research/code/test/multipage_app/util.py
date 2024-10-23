import os
import streamlit as st
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

btn_labels = ["load data check", "duplicate check", "quality check", "metadata check", "fix metadata", "data loader check" ]
unpressed_colour = "#707070"
success_colour = "#4CAF50"
failure_color = "#FF7F50"
wip_color = "#FFD700"


def ChangeButtonColour(widget_label, prsd_status):
    btn_bg_colour = success_colour if prsd_status == True else unpressed_colour
    if prsd_status == 1:
        btn_bg_colour = success_colour
    elif prsd_status == 2:
        btn_bg_colour = failure_color
    else:
        btn_bg_colour = wip_color    
    htmlstr = f"""
        <script>
            var elements = window.parent.document.querySelectorAll('button');
            for (var i = 0; i < elements.length; ++i) {{ 
                if (elements[i].innerText == '{widget_label}') {{ 
                    elements[i].style.background = '{btn_bg_colour}'
                }}
            }}
        </script>
        """
    components.html(f"{htmlstr}", height=0, width=0)


def ChkBtnStatusAndAssignColour():
    for i in range(len(btn_labels)):
        ChangeButtonColour(btn_labels[i], mystate.btn_prsd_status[i])


def btn_pressed_callback(i,j):
    #mystate.btn_prsd_status = [False] * 6
    mystate.btn_prsd_status[i - 1] = j


with st.container():
    c1, c2, c3, c4, c5,c6 = st.columns ((1, 1, 1, 1, 1, 1), gap='small')
    c1.button("load data check", key="g1", on_click=btn_pressed_callback, args=(1,1))
    c2.button("duplicate check", key="g2", on_click=btn_pressed_callback, args=(2,3))
    c3.button("quality check", key="g3", on_click=btn_pressed_callback, args=(3,2))
    c4.button("metadata check", key="g4", on_click=btn_pressed_callback, args=(4,1))
    c5.button("fix metadata", key="g5", on_click=btn_pressed_callback, args=(5,2))
    c6.button("data loader check", key="g6", on_click=btn_pressed_callback, args=(6,2))
    ChkBtnStatusAndAssignColour() 