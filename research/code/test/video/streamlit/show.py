import streamlit as st

st.header("video display")

vf = open("/home/madhekar/Videos/ffmpeg_frames/video_1/VID_20181205_121309.mp4", "rb")

vb = vf.read()

st.video(vb)
