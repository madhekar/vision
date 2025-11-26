import streamlit as st
import pandas as pd
from streamlit_tags import st_tags
from streamlit_imagegrid import streamlit_imagegrid as image_grid

st.set_page_config(layout="wide")
st.title("Interactive Image Gallery with Tags")

# 1. Structure your image data
# This data would likely come from a database or file system in a real app
image_data = [
    {"url": "picsum.photos", "title": "Desk", "tags": ["office", "wood", "work"]},
    {"url": "picsum.photos", "title": "Chair", "tags": ["furniture", "wood"]},
    {"url": "picsum.photos", "title": "Building", "tags": ["city", "architecture"]},
    {"url": "picsum.photos", "title": "Road", "tags": ["city", "transport"]},
    # Add more images as needed
]
df = pd.DataFrame(image_data)

# Extract all unique tags for the filter options
all_tags = sorted(list(set([tag for tags in df['tags'] for tag in tags])))

# 2. Add interactive tag filter input
st.sidebar.header("Filter by Tags")
selected_tags = st.sidebar.multiselect(
    'Select tags to display images',
    options=all_tags,
    default=[]
)

# 3. Filter the DataFrame based on selected tags
if selected_tags:
    filtered_df = df[df['tags'].apply(lambda tags: all(tag in tags for tag in selected_tags))]
else:
    filtered_df = df

# Prepare data for streamlit-imagegrid (it expects a list of dictionaries)
# The 'metadata' field is used for the tags display
grid_data = filtered_df.rename(columns={'tags': 'metadata'}).to_dict('records')

st.write(f"Displaying {len(filtered_df)} images.")

# 4. Display the images in a grid
if not filtered_df.empty:
    # 'image_grid' handles the responsive grid layout and displays tags
    # Check the streamlit-imagegrid docs for options on selection and layout
    image_grid("Data",grid_data, 4, key='10') #"visualization1", urls, 4, key='foo'
else:
    st.info("No images match the selected tags.")

# You can also use the streamlit-tags component for a different input style
# keywords = st_tags(
#     label='Enter Keywords:',
#     text='Press enter to add more',
#     value=['one', 'two', 'three']
# )
