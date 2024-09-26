import asyncio
import random
import folium
import streamlit as st
from streamlit_folium import st_folium


def random_marker():
    random_lat = random.random() * 0.5 + 37.79
    random_lon = random.random() * 0.5 - 122.4
    return folium.Marker(
        location=[random_lat, random_lon],
        popup=f"Random marker at {random_lat:.2f}, {random_lon:.2f}",
    )


def add_random_marker():
    marker = random_marker()
    print(f'Adding marker {marker}')
    st.session_state["markers"].append(marker)


def clear_markers():
    st.session_state["markers"].clear()


async def add_markers_to_map_async():
    while True:
        marker = random_marker()
        print(f'Adding marker {marker}')
        st.session_state["markers"].append(marker)
        await asyncio.sleep(1)


async def main():
    if "markers" not in st.session_state:
        st.session_state["markers"] = []

    if st.button("Add random marker"):
        add_random_marker()

    if st.button("Clear markers"):
        clear_markers()

    if st.button("Start adding markers asynchronously"):
        await add_markers_to_map_async()

    m = folium.Map(location=[37.95, -122.200], zoom_start=10)
    fg = folium.FeatureGroup(name='Markers')
    for marker in st.session_state["markers"]:
        fg.add_child(marker)
    st_folium(m, feature_group_to_add=fg, width=725, key='user-map', returned_objects=[])


if __name__ == "__main__":
    asyncio.run(main())