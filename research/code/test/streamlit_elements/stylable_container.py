import streamlit as st
from streamlit_extras.stylable_container import stylable_container

# with stylable_container(
#         key="comic_sans_button",
#         css_styles=["""
#          button {
#             min-height: 8px;
#             min-width: 100px;       
#          }
#          """,
#          """
#          button > div >  p {
#             font-size: 7px; height: 8px; width: 100px; font-family: "Comic Sans MS", "Comic Sans", cursive;
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
            min-height: 8px ; /* Set the desired height */
            min-width: 30px;  /* Set the desired width */
            /*font-size: 6px; Optional: adjust font size for better fit */
        }
    """,
    """
    button > div >  p {
            font-size: 7px; height: 8px; width: 30px; font-family: "Comic Sans MS", "Comic Sans", cursive;
          }
          """
    ],
):
    st.button("Click Me!" )

st.button("Other button")