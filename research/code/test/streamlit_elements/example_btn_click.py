import streamlit as st
import streamlit.components.v1 as components
import random
import time

# Set page title
st.set_page_config(
    page_title="test",
    layout="wide",
)

mystate = st.session_state
if "btn_prsd_status" not in mystate:
    mystate.btn_prsd_status = [False] * 8

btn_labels = ["Button 1", "Button 2", "Button 3", "Button 4", "Button 5", "Button 6", "Button 7", "Button 8"]
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
    components.html(f"{htmlstr}")

def ChkBtnStatusAndAssignColour():
    for i in range(len(btn_labels)):
        ChangeButtonColour(btn_labels[i], mystate.btn_prsd_status[i])

def btn_pressed_callback(i):
    mystate.btn_prsd_status = [False] * 8
    mystate.btn_prsd_status[i-1] = True

with st.container():
    # Stremalit columns initialize for the buttons
    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(
        [1, 1, 1, 1, 1, 1, 1, 1]
    )

    # Button 1
    col1.button(
        "Button 1",
        key=None,
        help="help info",
        on_click=btn_pressed_callback,
        args=(1,),
        use_container_width=True,
    )

    # Button 2
    col2.button(
        "Button 2",
        key=None,
        help="help info",
        on_click=btn_pressed_callback,
        args=(2,),
        use_container_width=True,
    )

    # Button 3
    col3.button(
        "Button 3",
        key=None,
        help="help info",
        on_click=btn_pressed_callback,
        args=(3,),
        use_container_width=True,
    )

    # Button 4
    col4.button(
        "Button 4",
        key=None,
        help="help info",
        on_click=btn_pressed_callback,
        args=(4,),
        use_container_width=True,
    )

    # Button 5
    col5.button(
        "Button 5",
        key=None,
        help="help info",
        on_click=btn_pressed_callback,
        args=(5,),
        use_container_width=True,
    )

    # Button 6
    col6.button(
        "Button 6",
        key=None,
        help="help info",
        on_click=btn_pressed_callback,
        args=(6,),
        use_container_width=True,
    )

    # Button 7
    col7.button(
        "Button 7",
        key=None,
        help="help info",
        on_click=btn_pressed_callback,
        args=(7,),
        use_container_width=True,
    )

    # Button 8
    col8.button(
        "Button 8",
        key=None,
        help="help info",
        on_click=btn_pressed_callback,
        args=(8,),
        use_container_width=True,
    )

    # Check button status and color
    ChkBtnStatusAndAssignColour()

    st.markdown(
        """
        <style>
           div[data-testid="stHorizontalBlock"] { position:fixed; bottom:126.5px; width: 89.3%; text-align: 
                center; padding-right: 9%; padding-left: 9%; z-index: 99; opacity: 1; background-color: #ece1ec;}
        </style>""",
        unsafe_allow_html=True,
    )

   
    
with st.container():
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            assistant_response = random.choice(
                [
                    "Hello there! How can I assist you today?",
                    "Hi, human! Is there anything I can help you with?",
                    "Do you need help?",
                ]
            )
            # Simulate stream of response with milliseconds delay
            for chunk in assistant_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

