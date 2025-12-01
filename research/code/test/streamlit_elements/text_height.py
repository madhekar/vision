import streamlit as st
#text_area_custom_height = st.text_area("Custom Height Text Area", height=0)


# custom_css = """
# <style>
# textarea.st-cl {
#     height: 120px;
#     font-size: 8px;
# }
# </style>
# """
# st.markdown(custom_css, unsafe_allow_html=True)
# user_input = st.text_area("Type your text here:")

'''
.stTextInput input[aria-label="Address"] {
    field-sizing: content;
    background-color: #ffffe0; /* LightYellow */
    border: 0px solid #ffebcd; /* BlanchedAlmond */
    color: #000000;
    height: 11px;
    line-height: 12px;
    font-size: 8px;
'''

custom_css = """
<style>
text_input.st-cl {
    height: 10px;
    width: 10px;
    font-size: 5px;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)
user_input = st.text_input("Type your text here:")