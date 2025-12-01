import streamlit as st
import streamlit_antd_components as sac

btn = sac.buttons(
    items=['button1', 'button2', 'button3'],
    index=0,
    format_func='title',
    align='center',
    size="xl",
    color="gold",
    direction='horizontal',
    radius='lg',
    return_index=False,
)

sac.text_input(
    size="sx"
)
st.write(f'The selected button label is: {btn}')