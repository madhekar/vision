import streamlit as st
from PIL import Image

def main():
    st.title("Image Grid Example")
    
    static_locations = (["san",(22.0,-110.9)], ["sfo",(56.99, -89.87)])
    image_metadata = [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
    # Load images
    image_paths = [
        "/home/madhekar/work/home-media-app/data/input-data/img/Collage1.jpg",
        "/home/madhekar/work/home-media-app/data/input-data/img/Collage2.jpg",
        "/home/madhekar/work/home-media-app/data/input-data/img/Collage3.jpg",
        "/home/madhekar/work/home-media-app/data/input-data/img/Collage4.jpg",
    ]
    index = st.selectbox("select location" ,range(len(static_locations)), format_func=lambda x: static_locations[x][0])
    st.write(index)
    images = [Image.open(path) for path in image_paths]
    
    # Create image grid
    cols = st.columns(3)  # 3 columns per row
    selected_images = []

    for i, image in enumerate(images):
        with cols[i % 3]:
            checkbox = st.checkbox("", key=f"checkbox_{i}")
            if checkbox:
                selected_images.append(image)
                image_metadata[i] = static_locations[index][1]
            st.image(image, use_column_width=True)

    # Display selected images
    if selected_images:
        st.subheader("Selected Images")
        for image in selected_images:
            st.image(image)

    st.write(image_metadata)
if __name__ == "__main__":
    main()