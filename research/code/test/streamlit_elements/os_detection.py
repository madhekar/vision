import streamlit as st
import platform
import os

st.title("Server OS Detector")

# Get OS information
os_name = os.name
platform_system = platform.system()
platform_release = platform.release()
platform_version = platform.version()

# Display information in Streamlit
st.write(f"**OS Name:** {os_name}")
st.write(f"**Platform System:** {platform_system}")
st.write(f"**Platform Release:** {platform_release}")
st.write(f"**Platform Version:** {platform_version}")

# Example of conditional logic based on OS
if platform_system == "Windows":
    st.success("Running on Windows")
elif platform_system == "Linux":
    st.success("Running on Linux (likely Streamlit Cloud/Docker)")
else:
    st.info(f"Running on: {platform_system}")
