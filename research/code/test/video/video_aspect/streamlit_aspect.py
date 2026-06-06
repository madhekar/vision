import streamlit as st
from streamlit_js_eval import streamlit_js_eval

def display(vid_input,vid_modified, ht):
    co, cm = st.columns([1,1])

    with co:
        with st.container(key="my_custom_container"):#(height=500, border=False):
            #st.markdown('<div class="responsive-img-container">', unsafe_allow_html=True)
            video_file = open(vid_input, "rb")
            video_bytes = video_file.read()
            st.video(video_bytes)
            #st.markdown('</div>', unsafe_allow_html=True)
    with cm:
        #st.markdown('<div class="responsive-img-container">', unsafe_allow_html=True)
        st.write("----", ht)
        #ht = ht -100
        # with st.container(key="my_custom_container_image"):
        #     #st.markdown('<div class="dynamic-container">', unsafe_allow_html=True)
    
        #     video_file = open(vid_modified, "rb")
        #     video_bytes = video_file.read()
        #     st.video(video_bytes)
        #     st.markdown('</div>', unsafe_allow_html=True)



st.set_page_config(
    page_title="Media Portal (MP)",
    initial_sidebar_state="expanded",
    layout="wide"
)

st.html("""
    <style>
    .responsive-img-container img {
        width: 100% !important;
        height: 50px !important; /* Forces uniform height across row */
        object-fit: cover !important; /* Crops cleanly instead of squishing */
        border-radius: 8px;
    }
    </style>
""")

# import streamlit as st

# st.set_page_config(layout="wide")

# # Assign a key so we can style this specific container
# with st.container(key="my-vh-container", border=True):
#     st.write("This container will take up 80% of the screen height.")

# st.markdown(
#     """
#     <style>
#         .st-key-my_vh_container {
#             height: 80vh;
#             overflow-y: auto; /* Adds a scrollbar if content overflows */
#         }
#     </style>
#     """,
#     unsafe_allow_html=True
# )


st.markdown(
    """
    <style>
    div[class*="st-key-my_custom_container"] {
        max-height: 60vh !important;
        max-height: 60dvh !important;
        overflow-y: auto  !important; /* Adds a scrollbar if content overflows */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    div[class*="st-key-my_custom_container_image"] {
        max-height: 80vh !important;
        max-height: 80dvh !important;
        overflow-y: auto !important; /* Adds a scrollbar if content overflows */
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    .dynamic-container {
        max-height: 80vh; 
        overflow-y: auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

o_vidio = "/home/madhekar/tmp/VID_20181205_171018.mp4"
s_video = "/home/madhekar/tmp/output_1080p_video.mp4"
st.title("video comparision")    

height = streamlit_js_eval(js_expressions='screen.height', key='SCR')
st.write(f"height: {height}")

# cw= None# st.get_container_width
# sw = st.screen_width

# st.write(cw,"::" ,sw)
display(o_vidio, s_video, height)

