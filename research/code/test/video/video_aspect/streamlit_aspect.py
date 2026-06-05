import streamlit as st


def display(vido,vidm):
    co, cm = st.columns([1,1])

    with co:
        video_file = open(vido, "rb")
        video_bytes = video_file.read()
        st.video(video_bytes)

    with cm:
        video_file = open(vidm, "rb")
        video_bytes = video_file.read()
        st.video(video_bytes)


st.set_page_config(
    page_title="zesha: Media Portal (MP)",
    initial_sidebar_state="expanded",
    layout="wide"
)
o_vidio = "/home/madhekar/tmp/VID_20181205_171018.mp4"
s_video = "/home/madhekar/tmp/output_1080p_video.mp4"
st.title("video comparision")    

display(o_vidio, s_video)

