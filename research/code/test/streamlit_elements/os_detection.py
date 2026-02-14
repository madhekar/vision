import streamlit as st
import platform
from pathlib import Path, WindowsPath
import os

img_path = "/home/madhekar/work/home-media-app/data/final-data/img/madhekar/2596441a-e02f-588c-8df4-dc66a133fc99/IMG_6683.PNG"
linux_prefix = "/mnt/zmdata/"
mac_prefix = "/Users/Share/zmdata/"
win_prefix = "c:/Users/Public/zmdata/"
token = "home-media-app"
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
    w_img_path = WindowsPath(img_path)
    parts = w_img_path.split(token,1)
    if len(parts) > 1:
        n_pth = win_prefix + token + parts[1]
    st.success(f"Running on Windows: {n_pth} ")

elif platform_system == "Linux":
    parts = img_path.split(token,1)
    if len(parts) > 1:
        n_pth = linux_prefix + token + parts[1]
    st.success(f"Running on Linux: {n_pth}")

elif platform_system == "Darwin":
    parts = img_path.split(token,1)
    print(parts)
    if len(parts) > 1:
        n_pth = mac_prefix + token + parts[1]
    st.success(f"Running on MacOS : {n_pth}")
else:    
    st.info(f"Running on: {platform_system}")
