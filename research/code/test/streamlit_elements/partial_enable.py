import streamlit as st

# # Track or check a value in session state or a first selectbox
# box1 = st.selectbox("Choose the first option", ["Enable Box 2", "Disable Box 2"], index=0)

# # Disable the second selectbox if 'Disable Box 2' is selected
# is_disabled = box1 == "Disable Box 2"

# box2 = st.selectbox(
#     "Choose the second option",
#     ["Option A", "Option B", "Option C"],
#     disabled=is_disabled,
# )




def custom_selectbox(label, options, disabled_options):
    selected_option = st.selectbox(label, options)
    if selected_option in disabled_options:
        st.markdown(f"Selected option: **{selected_option}** (disabled)")
    else:
        st.markdown(f"Selected option: **{selected_option}**")

options = ["Option 1", "Option 2", "Option 3"]
disabled_options = ["Option 2"]  # List of options you want to disable

custom_selectbox("Select an option", options, disabled_options)