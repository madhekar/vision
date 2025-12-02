import streamlit as st


st.markdown("""
    <style>
        /* Targeting the nested structure */
         div p {
            color: blue;
            font-size: 8px;
            font-weight: bold;
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Your Streamlit content that creates the nested divs and paragraphs
# For demonstration purposes, let's create a simple nested structure
with st.container():
    st.write("Outer container")
    with st.container():
        st.write("Second level container")
        with st.container():
            st.write("Third level container")
            with st.container():
                st.write("Fourth level container")
                st.write("This is a paragraph inside the nested divs.")
# st.markdown("""
#     <style>
#         .my-nested-paragraph {
#             color: green;
#             font-size: 20px;
#             text-decoration: underline;
#         }
#     </style>
# """, unsafe_allow_html=True)

# with st.container():
#     st.markdown("<div><div><div><div><p class='my-nested-paragraph'>This paragraph is styled with a class.</p></div></div></div></div>", unsafe_allow_html=True)
