import streamlit as st
import streamlit.components.v1 as components


mystate = st.session_state
if "btn_prsd_status" not in mystate:
    mystate.btn_prsd_status = [False] * 6

btn_labels = ["load data check", "de-duplication check", "quality check", "missing metadata check", "fix Metadata check", "metadata loader check" ]
unpressed_colour = "#707070"
pressed_colour = "#4CAF50"


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
    mystate.btn_prsd_status = [False] * 6
    mystate.btn_prsd_status[i - 1] = True


with st.container():
    c1, c2, c3, c4, c5,c6 = st.columns((1, 1, 1, 1, 1, 1))
    c1.button("load data check", key="b1", on_click=btn_pressed_callback, args=(1,))
    c2.button("de-duplication check", key="b2", on_click=btn_pressed_callback, args=(2,))
    c3.button("quality check", key="b3", on_click=btn_pressed_callback, args=(1,))
    c4.button("missing metadata check", key="b4", on_click=btn_pressed_callback, args=(2,))
    c5.button("fix Metadata check", key="b5", on_click=btn_pressed_callback, args=(1,))
    c6.button("metadata loader check", key="b6", on_click=btn_pressed_callback, args=(2,))
    ChkBtnStatusAndAssignColour() 