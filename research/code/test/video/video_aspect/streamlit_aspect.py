import streamlit as st


def display(vid_input,vid_modified):
    co, cm = st.columns([1,1])

    with co:
        st.markdown('<div class="responsive-img-container">', unsafe_allow_html=True)
        video_file = open(vid_input, "rb")
        video_bytes = video_file.read()
        st.video(video_bytes)
        st.markdown('</div>', unsafe_allow_html=True)
    with cm:
        st.markdown('<div class="responsive-img-container">', unsafe_allow_html=True)
        video_file = open(vid_modified, "rb")
        video_bytes = video_file.read()
        st.video(video_bytes)
        st.markdown('</div>', unsafe_allow_html=True)

st.html("""
    <style>
    .responsive-img-container img {
        width: 100% !important;
        height: 250px !important; /* Forces uniform height across row */
        object-fit: cover !important; /* Crops cleanly instead of squishing */
        border-radius: 8px;
    }
    </style>
""")

st.set_page_config(
    page_title="Media Portal (MP)",
    initial_sidebar_state="expanded",
    layout="wide"
)

o_vidio = "/home/madhekar/tmp/VID_20181205_171018.mp4"
s_video = "/home/madhekar/tmp/output_1080p_video.mp4"
st.title("video comparision")    

display(o_vidio, s_video)

