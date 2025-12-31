import streamlit as st

col1, col2, col3 = st.columns([1, 0.1, 1]) # Adjust column ratios as needed

with col1:
    st.header("Left Column")
    st.write("Content goes here.")

with col2:
    # Use st.html to inject custom CSS for a vertical line
    st.write('b h a l c h a n d r a')
    st.html(
        """
        <style>
        .vertical-divider {
            border-left: 2px solid #e6e6e6; # Adjust color and thickness
            height: 100%;                  # Ensure it spans the full height
            margin: 0 auto;                 # Center the divider in the column
        }
        </style>
        <div class="vertical-divider"></div>
        """
    )
    st.write("- - - - - - - -")

with col3:
    st.header("Right Column")
    st.write("More content here.")