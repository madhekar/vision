import streamlit as st

st.set_page_config(layout="wide")

# Inject CSS to force uniform image wrapper sizing and handle cropping cleanly
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

cols = st.columns(3)

for col in cols:
    with col:
        # Wrap image in a div targeting the custom CSS class
        st.markdown(
            '<div class="responsive-img-container">', 
            unsafe_allow_html=True
        )
        st.image("/mnt/zmdata/home-media-app/data/input-data/img/madhekar/74eef5fc-ffae-5e08-b3bf-62964dd279e7/IMG_9479.PNG")
        st.markdown('</div>', unsafe_allow_html=True)
