import streamlit as st
from streamlit_extras.stylable_container import stylable_container

# Define the options for the selectbox
options = ["Red", "Blue", "Green", "Yellow"]

# Use stylable_container to style the selectbox
with stylable_container(
    key="styled_selectbox_container",
    css_styles="""
    .element-container {
        background-color: #f0f2f6;
        padding: 1vw;
        width: 5vw;
        height: 1vw;
        border-radius: 1vw;
        box-shadow: 0 1vw 2vw rgba(0,0,0,0.1);
        font-size: 3vw;
    }
    .element-container:hover {
        background-color: #e0e2e6;
    }
    """
):
    # Create the selectbox inside the styled container
    selected_color = st.selectbox(
        'Choose your favorite color',
        options,
        key="my_selectbox"
    )

# Display the selected value
#st.write('You selected:', selected_color)