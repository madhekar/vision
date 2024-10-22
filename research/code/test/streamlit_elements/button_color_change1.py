import streamlit as st
import streamlit.components.v1 as components

mystate = st.session_state
if "btn_prsd_status" not in mystate:
    mystate.btn_prsd_status = [False] * 2

btn_labels = ["General", "Whole Engine"]
unpressed_colour = "#E8EAF6"
pressed_colour = "#64B5F6"

def ChangeButtonColour(widget_label, prsd_status):
    btn_bg_colour = pressed_colour if prsd_status == True else unpressed_colour
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

def btn_pressed_callback(i):
    mystate.btn_prsd_status = [False] * 2
    mystate.btn_prsd_status[i-1] = True

with st.container():
    col1, col2 = st.columns((1, 1))
    col1.button("General", key="b1", on_click=btn_pressed_callback, args=(1,) )
    col2.button("Whole Engine", key="b2", on_click=btn_pressed_callback, args=(2,) )
    ChkBtnStatusAndAssignColour()
