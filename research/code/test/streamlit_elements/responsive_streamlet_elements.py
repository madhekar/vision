import streamlit as st

st.markdown("""
    <style>
        /* Your custom CSS rules here */
        .responsive-container {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }
        .responsive-item {
            flex: 1 1 200px; /* Grow, shrink, and base width */
            background-color: lightblue;
            padding: 2px;
            text-align: center;
        }
        @media (max-width: 1024px) {
            .responsive-item {
                flex: 1 1 100%; /* Full width on smaller screens */
            }
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="responsive-container">', unsafe_allow_html=True)
st.markdown('<div class="responsive-item">Item 1</div>', unsafe_allow_html=True)
st.markdown('<div class="responsive-item">Item 2</div>', unsafe_allow_html=True)
st.markdown('<div class="responsive-item">Item 3</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
