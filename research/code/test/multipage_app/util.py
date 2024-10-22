import streamlit as st
import streamlit.components.v1 as components


def ChangeButtonColour(wgt_txt, wch_hex_colour="12px"):
    htmlstr = (
        """<script>var elements = window.parent.document.querySelectorAll('*'), i;
                for (i = 0; i < elements.length; ++i) 
                    { if (elements[i].innerText == |wgt_txt|) 
                        { elements[i].style.color ='"""
        + wch_hex_colour
        + """'; } }</script>  """
    )

    htmlstr = htmlstr.replace("|wgt_txt|", "'" + wgt_txt + "'")
    components.html(f"{htmlstr}", height=0, width=0)


if __name__=='__main__':
    cols = st.columns(4)
    cols[0].button("no colour", key="b1")
    cols[1].button("green", key="b2")
    cols[2].button("red", key="b3")
    cols[3].button("no colour", key="b4")

    ChangeButtonColour("green", "#4E9F3D")  # button txt to find, colour to assign
    ChangeButtonColour("red", "#FF0000")  # button txt to find, colour to assign  