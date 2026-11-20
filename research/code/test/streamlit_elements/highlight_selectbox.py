import streamlit as st

options = ["Option 0 (Default)", "Option 1 (Highlighted)", "Option 2", "Option 3","Option 4","Option 5"]
target_index = 1  # 0-based Python index
css_index = target_index + 1  # 1-based CSS nth-child index

# Inject CSS to color the specific option in the dropdown list view
st.markdown(
    f"""
    <style>
    /* Target the specific item by index in the open dropdown list */
    div[data-baseweb="popover"] ul[role="listbox"] li:nth-child({css_index}) {{
        background-color: #FF0000 !important;
        color: #FF0000 !important;
        font-weight: bold;
    }}

    /* Optional: Custom hover color for that specific item */
    div[data-baseweb="popover"] ul[role="listbox"] li:nth-child({css_index}):hover {{
        background-color: #ff0000 !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

selected = st.selectbox("Choose an item:", options)
