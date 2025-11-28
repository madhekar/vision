import streamlit as st

tabs_font_css = """
<style>
div[class*="stTextArea"] label p {
  font-size: 26px;
  color: red;
}

div[class*="stTextInput"] label p {
  font-size: 26px;
  color: blue;
}



div[class*="stNumberInput"] label p {
  font-size: 26px;
  color: green;
}
</style>
"""

st.write(tabs_font_css, unsafe_allow_html=True)

st.text_area("Text area")
st.text_input("Text input")
st.number_input("Number input")