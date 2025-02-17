import streamlit as st
import time

def createProgressBar(pg_caption, pg_int_percentage, pg_colour, pg_bgcolour):
    pg_int_percentage = str(pg_int_percentage).zfill(2)
    pg_html = f"""<table style="width:50%; border-style: none;">
                        <tr style='font-weight:bold;'>
                            <td style='background-color:{pg_bgcolour};'>{pg_caption}: <span style='accent-color: {pg_colour}; bgcolor: transparent;'>
                                <progress value='{pg_int_percentage}' max='100'>{pg_int_percentage}%</progress> </span>{pg_int_percentage}% 
                            </td>
                        </tr>
                    </table><br>"""
    return pg_html

latest_iteration = st.sidebar.empty()
bar = st.sidebar.progress(0)
sidebar= st.sidebar.markdown(createProgressBar("Positive", 0, "#A5D6A7", "#B2EBF2"), True)
num = 20
for i in range(num):
    latest_iteration.text(f'{num - (i + 1)} seconds left')
    bar.progress((100//num)*i)
    sidebar.progress(i)
    time.sleep(1)