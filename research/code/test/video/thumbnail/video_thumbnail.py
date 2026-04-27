'''
To create video thumbnails using Python, you can use specialized video processing libraries like MoviePy or the high-performance command-line tool FFmpeg via Python wrappers. 
1. Using MoviePy (Easiest)
MoviePy is a popular library for video editing that allows you to extract a frame at a specific timestamp with just a few lines of code. 

    Installation: pip install moviepy
    Code Example:
    python

    from moviepy.editor import VideoFileClip

    # Load the video
    clip = VideoFileClip("video.mp4")

    # Save the frame at 5 seconds as an image
    clip.save_frame("thumbnail.jpg", t=5.0) 

    Use code with caution.
     

2. Using FFmpeg (Most Versatile)
For higher performance or more complex tasks, FFmpeg is the industry standard. You can call it from Python using the subprocess module or libraries like ffmpeg-python. 

    Installation: Ensure FFmpeg is installed on your system.
    Python Subprocess Command:
    python

    import subprocess

    # -ss: timestamp, -i: input file, -vframes 1: capture one frame
    subprocess.call(['ffmpeg', '-i', 'video.mp4', '-ss', '00:00:05.000', '-vframes', '1', 'thumbnail.jpg'])

    Use code with caution.
     

%%MAGIT_PARSER_PROTECT%% ``` 
3. Using OpenCV (For Frame Analysis) 
If you need to analyze frames (e.g., to find a non-black frame), OpenCV is the best choice. 

    Installation: pip install opencv-python
    Code Example:
    %%MAGIT_PARSER_PROTECT%% ```python
    import cv2vcap = cv2.VideoCapture("video.mp4")
    Read a specific frame (e.g., frame 100)
    vcap.set(cv2.CAP_PROP_POS_FRAMES, 100)
    success, image = vcap.read()
    if success:
    cv2.imwrite("thumbnail.png", image)
    %%MAGIT_PARSER_PROTECT%% ``` 

4. Specialized Tools

    pyvideothumbnailer: A command-line tool and Python package specifically designed to create "contact sheet" thumbnails (grids of multiple frames).
    thumbnails: A modern library for generating thumbnails for web video players, supporting WebVTT and JSON outputs. 

Do you need to create a single snapshot at a specific time, or are you looking to generate a grid/contact sheet of multiple frames?

'''

'''
Streamlit clickable videos can be achieved by embedding video links within markdown, using st.video for playback, or using custom components to detect clicks on video thumbnails. Key approaches include rendering HTML in markdown (unsafe_allow_html=True), using image components (st-clickable-images) to launch videos in a new tab, or triggering callbacks upon interaction. 
Here are the primary methods for making clickable videos in Streamlit:

    Embed Clickable Links in Markdown: Use st.markdown with <a href="..."> to make a video thumbnail or link clickable.
    Use st.video: The st.video function natively supports YouTube URLs or local paths for playback.
    Custom Components: Utilize components like st-clickable-images (from pip install st-clickable-images) to display thumbnails and detect which one was clicked.
    Create Clickable Dataframe Cells: Render interactive HTML links within Pandas DataFrames using st.write and to_html(escape=False). 

Example Implementation (Clickable Thumbnail):
python

import streamlit as st

# Display a clickable image that acts as a video link
st.markdown(
    """
    <a href="https://www.youtube.com/watch?v=your_video_id">
        <img src="https://img.youtube.com/vi/your_video_id/0.jpg" width="300">
    </a>
    """,
    unsafe_allow_html=True
)
st.write("Click the image above to watch the video.")

Use code with caution.
Key Considerations:

    unsafe_allow_html=True: Required in st.markdown to render HTML tags.
    st.session_state: Necessary to track which video was clicked if you want to update the app content dynamically. 

Would you like to know more about:

    Using callbacks to update the page when a video is clicked?
    Customizing the layout using columns (st.columns) for multiple videos?
    Using st-click-detector for more complex interactive elements? 
'''

import subprocess

def create_thumb(input_path, output_path):
    command = [
        'ffmpeg',
        '-ss', '00:00:10', # Seek to 10 seconds
        '-i', input_path,
        '-vframes', '1',
        '-q:v', '2',       # Image quality (2-5 is high)
        output_path
    ]
    subprocess.run(command)

create_thumb('input.mp4', 'output.jpg')
