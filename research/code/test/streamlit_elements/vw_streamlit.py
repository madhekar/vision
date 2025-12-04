import streamlit as st

st.markdown(
    """
    <style>
    .my-responsive-text {
        font-size: 3vw; /* Font size will be 3% of the viewport width */
    }
    .my-responsive-container {
        width: 80vw; /* Container width will be 80% of the viewport width */
        margin: auto; /* Center the container */
        border: .1vw solid blue;
        padding: 1vw;
    }
    .my-responsive-button {
      font-size: 3vw;
      height: 4vw; 
      width: 10vw;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Streamlit with VW Units")
st.markdown("<p class='my-responsive-text'>This text scales with viewport width.</p>", unsafe_allow_html=True)

with st.container():
    st.markdown("<div class='my-responsive-container'>This container also scales.</div>", unsafe_allow_html=True)

    st.markdown("<div class='my-responsive-button'> ggg</div>", unsafe_allow_html=True)
