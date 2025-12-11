import streamlit as st

# Custom CSS to control the width of ALL st.text_input widgets
st.markdown("""
<style>
/* Target the main container for text input widgets */
[data-testid="stTextInput"] {
    #width: 300px; /* Set a fixed width in pixels */
    /* Or set a specific percentage of the parent container's width */
    width: 40%; 
    #height: .7rem;        
}

/* Optional: Target the actual input element inside the container */
[data-testid="stTextInput"] input {
    /* You can also adjust other properties like font size or color here */
    font-size: 1.2rem;
}
</style>
""", unsafe_allow_html=True)

st.text_input("Normal Width (but now fixed by CSS)")

# If you want a specific input to have a different style, 
# you can use a unique key and target it with an attribute selector.
st.markdown("""
<style>
    /* Target only the input with the key "short_input" */
    input[aria-label="Short Input"] {
        width: 50%;
        min-height: 8px;
        height: 9px;
        font-size: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.text_input("Short Input", key="short_input", help="This input is 50px wide")