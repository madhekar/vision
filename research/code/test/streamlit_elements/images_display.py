import streamlit as st
import time
import os
"""
In Streamlit, you can clear elements from the screen and write new ones by using
st.empty() as a placeholder that you can replace dynamically, or by utilizing st.session_state and st.rerun() to manage the application's flow and clear inputs. 
Method 1: Using st.empty() for Dynamic Replacement
The st.empty() method creates a single container in your app that can be cleared and rewritten with new elements as needed, without affecting the rest of the page. 
Using st.empty() allows you to create a placeholder that can be updated with new content. 
Method 2: Using st.session_state and st.rerun() for full page control
For managing the application's flow and state, especially for clearing inputs or forms, you can use st.session_state and st.rerun() to trigger a full page refresh. Clearing specific input widgets can be achieved by updating their values in st.session_state via a callback function. 
"""

def get_image_files(folder_path):
    """
    Returns a list of image filenames from a specified folder.
    """
    image_files = []
    valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']

    for filename in os.listdir(folder_path):
        # Get the file extension
        _, ext = os.path.splitext(filename)
        # Check if the extension is in the list of valid image extensions
        if ext.lower() in valid_extensions:
            filename = os.path.join(folder_path, filename)
            image_files.append(filename)
    return image_files



images = get_image_files("/home/madhekar/temp/travel/world/images")
print(images)

image_holder = st.empty()
for item in images:
    image_holder.image(item)
    time.sleep(1)