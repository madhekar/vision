import streamlit as st

st.set_page_config(layout="wide") # Optional: use the full screen width by default

# Inject custom CSS with a media query
st.markdown(
    """
    <style>
    /* Default style for input font size on larger screens */
    # input {
    #     font-size: 1rem !important;
    # }

    /* Media query for screens smaller than 600px */
    @media (max-width: 768px) {
        input {
            font-size: .6rem !important; /* Increase font size on mobile for better readability */
        }
    }

    /* Media query for screens larger than 900px */
    @media (min-width: 769px) and (max-width: 1024px) {
        input {
            font-size: 1.2rem !important; /* Increase font size on mobile for better readability */
        }
    }

    @media (min-width: 1025px) {
        input {
            font-size: 1.6rem !important; /* Increase font size on mobile for better readability */
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Responsive Text Input Example")

# Use columns for layout control (optional, but a common practice for responsiveness)
col1, col2 = st.columns([2, 1])

with col1:
    user_input = st.text_input("Enter some text here (font size changes on mobile)", key="responsive_input")

with col2:
    st.write(f"Current input: {user_input}")

st.write("Resize your browser window to see the font size change in the input box.")
