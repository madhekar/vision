import streamlit as st

st.set_page_config(page_title="Responsive REM Demo", layout="wide")

custom_css = """
<style>
/* Ensure the root font size can be adjusted by the browser */
html {
    font-size: 16px; /* Default font size for rem calculation */
}

/* Example of a responsive font size using a media query */
@media (max-width: 600px) {
    html {
        font-size: 14px; /* Smaller base font size on small screens */
    }
}

.my-text {
    font-size: 1.5rem; /* This will be 24px normally, or 21px on small screens */
    color: blue;
}

/* Targeting a Streamlit specific container for max width control (check dev tools for current selector) */
section[data-testid="stMain"] > div {
    max-width: 1200px;
    padding: 2rem; /* Responsive padding */
}
</style>
"""

# Use st.html for better rendering of raw HTML/CSS
st.html(custom_css)

st.markdown('<p class="my-text">This text uses 1.5rem and should change size with screen width.</p>', unsafe_allow_html=True)
st.button("A native Streamlit button")
