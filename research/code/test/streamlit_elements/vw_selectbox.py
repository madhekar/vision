import streamlit as st

# Define custom CSS with vw units
# This targets all selectboxes using their data-testid
custom_css = """
<style>
/* Target the main container of the selectbox widget */
div[data-testid="stSelectbox"] {
    width: 30vw !important; /* Set width to 30% of the viewport width */
    margin: 1vw auto; /* Center the selectbox and use 1% viewport margin */
    border: 1px solid #4CAF50; /* Add a border for visibility */
    border-radius: .5vw;
    padding: 0.5vw;
}

/* Target the actual selection display area */
div[data-baseweb="select"] div {
    font-size: 1.5vw; /* Adjust font size based on viewport width */
}
</style>
"""

# Inject the custom CSS into the Streamlit app
st.markdown(custom_css, unsafe_allow_html=True)

# Create the dropdown list (selectbox)
option = st.selectbox(
    "Choose a method:", # Label for the dropdown
    ("Email", "Home phone", "Mobile phone"), # Options in the list
    index=None, # Start with no option selected
    placeholder="Select contact method...", # Placeholder text
    key="contact_method_selectbox" # Key to target with CSS if needed for specificity (v1.39+)
)

# Display the selected option
st.write("You selected:", option)
