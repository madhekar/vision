import streamlit as st

# st.set_page_config(layout="wide")

# # Function to inject CSS
# def local_css(css_code):
#     st.markdown(f"<style>{css_code}</style>", unsafe_allow_html=True)

# # Create a slider to control the font size dynamically
# font_size_value = st.slider("Select Font Size (px) for Text Area Content", 10, 30, 16)
# label_size_value = st.slider("Select Font Size (px) for Text Area Label", 10, 30, 14)

# # Define the CSS rules based on slider values
# css_style = f"""
# /* Target the text content inside the text area */
# .stTextArea textarea {{
#     font-size: {font_size_value}px !important;
#     height: {font_size_value}px !important;
# }}

# /* Target the label of the text area */
# .stTextArea label p {{
#     font-size: {label_size_value}px !important;
#     height: {font_size_value}px !important;
# }}
# """

# # Inject the dynamic CSS
# local_css(css_style)

# # The text_area widget
# st.text_area(
#     label="Dynamically Sized Text Area",
#     value="Watch the font size of this text change dynamically as you move the sliders above! Use the Streamlit documentation to explore more styling options."
# )



# Function to inject custom CSS for font size and height
def set_style(height_px, size_px):
    custom_css = f"""
    <style>
        .stTextArea textarea {{
            min-height: 2px; !important;
            height: {height_px}px !important;
            font-size: {height_px}px !important;
        }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

# Use input widgets to control values dynamically
height_value = st.slider("Set height (px)", 5, 15, 1, key="height_slider")
font_size_value = st.number_input("Set font size (px)", 10, 30, 16, key="font_size_input")

set_style(height_value, font_size_value)
st.text_area("Dynamically Styled Text Area", value="Height and font size are controlled dynamically.")
