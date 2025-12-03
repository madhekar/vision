import streamlit as st
from streamlit_extras.stylable_container import stylable_container

mh, mw, fs, h, w = 8, 100, 7, 10, 100

# with stylable_container(
#         key="comic_sans_button",
#         css_styles=["""
#          button {
#             min-height: {mh}vw;
#             min-width: {mw}vw;       
#          }
#          """,
#          """
#          button > div >  p {
#             font-size: {fs}vw; height: {h}vw; width: {w}vw; font-family: "Comic Sans MS", "Comic Sans", cursive;
#          }
#          """
#          ],
# ):
#     st.button("Comic Sans button")
with stylable_container(
    key="button_container",
    css_styles=[
        """
        button {
            min-height: {mh}px ; /* Set the desired height */
            min-width: {mw}px;  /* Set the desired width */
            /*font-size: 6px; Optional: adjust font size for better fit */
        }
    """,
    """
    button > div >  p {
            font-size: {fs}px; height: {h}px; width: {w}px; font-family: "Comic Sans MS", "Comic Sans", cursive;
          }
          """
    ],
):
    st.button("Click Me!" )

st.button("Other button")