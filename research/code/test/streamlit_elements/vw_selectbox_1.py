import streamlit as st

st.markdown(
    """
    <style>
    div[data-baseweb="select"] .st-bd { /* Target the options container */
        width: 30vw !important; /* Set width to 50% of viewport width */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

options = ["Option A", "Option B", "Option C Long Text"]
selected_option = st.selectbox("Choose an option:", options)
st.write(f"You selected: {selected_option}")
