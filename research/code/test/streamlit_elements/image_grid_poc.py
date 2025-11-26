import streamlit_imagegrid
import streamlit as st

# List of image and video URLs
urls = [
    'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerEscapes.mp4',
    'https://images.freeimages.com/images/large-previews/56d/peacock-1169961.jpg',
    'https://images.freeimages.com/images/large-previews/bc4/curious-bird-1-1374322.jpg',
    'https://images.freeimages.com/images/large-previews/9f9/selfridges-2-1470748.jpg',
    'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4'
] * 2

# Generate the interactive grid with 4 columns
selected = streamlit_imagegrid.streamlit_imagegrid("visualization1", urls, 4, key='foo')

# Display the selected item
st.write(f"Selected item: {selected}")