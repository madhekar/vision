import streamlit as st
from PIL import Image

def main():
    st.title("Image Grid Example")

    # Load images
    image_paths = ["path/to/image1.jpg", "path/to/image2.jpg", "path/to/image3.jpg", "path/to/image4.jpg"]
    images = [Image.open(path) for path in image_paths]

    # Create image grid
    cols = st.columns(3)  # 3 columns per row
    selected_images = []

    for i, image in enumerate(images):
        with cols[i % 3]:
            checkbox = st.checkbox("", key=f"checkbox_{i}")
            if checkbox:
                selected_images.append(image)
            st.image(image, use_column_width=True)

    # Display selected images
    if selected_images:
        st.subheader("Selected Images")
        for image in selected_images:
            st.image(image)

if __name__ == "__main__":
    main()