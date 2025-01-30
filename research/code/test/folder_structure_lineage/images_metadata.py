import streamlit as st
from PIL import Image

def main():
    st.title("Image Grid Example")
    
    static_locations = [["san",22.0,-110.9], ["sfo",56.99, -89.87]]
    image_metadata = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    # Load images
    image_paths = [
        "/home/madhekar/work/home-media-app/data/input-data/img/Collage1.jpg",
        "/home/madhekar/work/home-media-app/data/input-data/img/Collage2.jpg",
        "/home/madhekar/work/home-media-app/data/input-data/img/Collage3.jpg",
        "/home/madhekar/work/home-media-app/data/input-data/img/Collage4.jpg",
    ]
    sel = st.selectbox(label='select static location', options=[i[0] for i in static_locations])
    print(sel)
    images = [Image.open(path) for path in image_paths]
    
    # Create image grid
    cols = st.columns(3)  # 3 columns per row
    selected_images = []

    for i, image in enumerate(images):
        with cols[i % 3]:
            checkbox = st.checkbox("", key=f"checkbox_{i}")
            if checkbox:
                selected_images.append(image)
                image_metadata[i] = sel
            st.image(image, use_column_width=True)

    # Display selected images
    if selected_images:
        st.subheader("Selected Images")
        for image in selected_images:
            st.image(image)

if __name__ == "__main__":
    main()