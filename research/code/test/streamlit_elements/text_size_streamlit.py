import streamlit as st

tabs_font_css = """
<style>
div[class*="stTextArea"] label p {
  font-size: .5rem;
  color: red;
}

div[class*="stTextInput"] label p {
  font-size: .4rem;
  color: blue;
}



div[class*="stNumberInput"] label p {
  font-size: .3rem;
  color: green;
}
</style>
"""

st.write(tabs_font_css, unsafe_allow_html=True)

st.text_area("Text area")
st.text_input("Text input")
st.number_input("Number input")