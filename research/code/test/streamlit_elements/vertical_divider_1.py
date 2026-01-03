import streamlit as st

# # 1. Define the custom CSS
# st.markdown("""
# <style>
# .single-border {
#     border-left: 15px solid red;
#     padding-left: 10px; /* Add some padding so content doesn't touch the border */
# }
# </style>
# """, unsafe_allow_html=True)

# # 2. Wrap content within an HTML div with the custom class
# st.markdown('<div class="single-border">', unsafe_allow_html=True)

# # 3. Place your Streamlit components inside the container
# st.subheader("Container Title")
# st.write("This container has a border only on the left side.")
# st.button("A button")

# # 4. Close the HTML div
# st.markdown('</div>', unsafe_allow_html=True)



st.write("Container below has only a left border:")

# 1. Create a container with a unique key
with st.container(key="my_container"):
    st.header("Container Content")
    st.write("This container has a border only on the left side.")
    st.button("Example Button")

# 2. Apply custom CSS targeting the key
st.markdown("""
<style>
/* Target the div that wraps the container using its key */
div[data-testid="stVerticalBlock"] > div:has(div.st-key-my_container) {
    border-left: 5px solid #FF4B4B; /* Example: a 5px solid red border */
    padding-left: 10px; /* Add some padding so content doesn't touch the border */
    border-radius: 0px; /* Optional: remove default rounded corners if present */
}
</style>
""", unsafe_allow_html=True)

st.write("This content is outside the styled container.")
