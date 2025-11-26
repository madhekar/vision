import streamlit as st
from streamlit_imagegrid import streamlit_imagegrid as image_grid

st.title("Streamlit Image Grid with Metadata")

# Prepare a list of media URLs with metadata and tags
urls = [
    {
        "src": "https://images.freeimages.com/images/large-previews/56d/peacock-1169961.jpg",
        "metadata": {"title": "Peacock", "artist": "John Doe"},
        "tags": {"animal": "bird", "color": "blue"},
    },
    {
        "src": "https://images.freeimages.com/images/large-previews/bc4/curious-bird-1-1374322.jpg",
        "metadata": {"title": "Curious Bird", "location": "Forest"},
        "tags": {"animal": "bird", "mood": "curious"},
    },
    {
        "src": "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerEscapes.mp4",
        "metadata": {"title": "For Bigger Escapes", "genre": "Action"},
        "tags": {"type": "video", "duration": "1m45s"},
    },
    {
        "src": "https://images.freeimages.com/images/large-previews/9f9/selfridges-2-1470748.jpg",
        "metadata": {"title": "Selfridges", "city": "London"},
        "tags": {"building": "store", "architecture": "modern"},
    },
]

# Display the image grid
clicked_item = image_grid("visualization1", urls, 4, key='foo')

if clicked_item:
    st.write(f"You clicked on: {clicked_item}")