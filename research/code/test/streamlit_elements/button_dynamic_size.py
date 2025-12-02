# Source - https://stackoverflow.com/a/76269909
# Posted by vvvvv
# Retrieved 2025-12-02, License - CC BY-SA 4.0

import streamlit as st

st.markdown(
    """
<style>
button {
    min-height: 20px;
    padding-top: 1px !important;
    padding-bottom: 1px !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# instructionCol, buttonCol = st.columns([4,1])
# with instructionCol:
#     with st.expander("Instructions"):
#         st.write("Pretend these are the instructions.")
# with buttonCol:
st.button("\nRestart\n") #, on_click=board.reset)
