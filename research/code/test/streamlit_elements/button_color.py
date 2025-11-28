import streamlit as st

""" ca, cb = st.columns([1,1])
with ca:
   print( st.get_container_width)
 """


from streamlit_js_eval import streamlit_js_eval
screen_width = streamlit_js_eval(label="screen.width",js_expressions='screen.width')
screen_height = streamlit_js_eval(label="screen.height",js_expressions='screen.height')
st.write(f"screen width: {screen_width} screen height: {screen_height}")
# import streamlit as st
# from streamlit_js_eval import streamlit_js_eval

st.write(f"Screen width: {streamlit_js_eval(js_expressions='screen.width', key='SCR')}")

st.button("Primary Button", type="primary")
st.button("Secondary Button", type="secondary") # Default
st.button("Tertiary Button", type="tertiary")