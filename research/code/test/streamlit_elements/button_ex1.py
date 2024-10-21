import streamlit as st


'''
https://docs.covalent.xyz/docs/user-documentation/tutorials/generativeai/?ref=blog.streamlit.io

'''
if 'stage' not in st.session_state:
    st.session_state.stage = 0

def set_state(i):
    st.session_state.stage = i

if st.session_state.stage >= 0:
    st.button('Data: Load RWA Data', on_click=set_state, args=[1])

if st.session_state.stage >= 1:
    name = st.button('Data: Filter Duplicate images', on_click=set_state, args=[2])  #text_input('Name', on_change=set_state, args=[2])

if st.session_state.stage >= 2:
    name = st.button('Data: Filter Bad quality images', on_click=set_state, args=[3])  #text_input('Name', on_change=set_state, args=[2])


# if st.session_state.stage >= 3:
#     st.write(f'Hello {name}!')
#     color = st.selectbox(
#         'Pick a Color',
#         [None, 'red', 'orange', 'green', 'blue', 'violet'],
#         on_change=set_state, args=[3]
#     )
#     if color is None:
#         set_state(2)

# if st.session_state.stage >= 4:
#     st.write(f':{color}[Thank you!]')
#     st.button('Start Over', on_click=set_state, args=[0])
