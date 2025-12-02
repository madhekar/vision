import streamlit as st


st.write("This is normal text")

st.toggle("Testing")

css = """
<style>
[data-baseweb="checkbox"] [data-testid="stWidgetLabel"] p {
    /* Styles for the label text for checkbox and toggle */
    font-size: .5rem;
    width: 30px;
    margin-top: .1rem;
}

[data-baseweb="checkbox"] div {
    /* Styles for the slider container */
    height: .3rem;
    width: .4rem;
}
[data-baseweb="checkbox"] div div {
    /* Styles for the slider circle */
    height: .28rem;
    width: .28rem;
}
[data-testid="stCheckbox"] label span {
    /* Styles the checkbox */
    height: .4rem;
    width: .4rem;
}
</style>
"""

if st.checkbox("Apply css", value=True):
    st.write(css, unsafe_allow_html=True)

    st.button("hi")