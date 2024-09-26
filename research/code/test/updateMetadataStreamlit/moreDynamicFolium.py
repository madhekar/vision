import random

import folium
import numpy as np
import streamlit as st
from folium.plugins import Draw, MeasureControl
from streamlit_folium import st_folium

st.set_page_config(page_title="st_folium Example", page_icon="🔎", layout="wide")
# Set up initial map state
CENTER_START = [37.8, -96]
ZOOM_START = 5


def initialize_session_state():
    if "center" not in st.session_state:
        st.session_state["center"] = CENTER_START
    if "zoom" not in st.session_state:
        st.session_state["zoom"] = ZOOM_START
    if "markers" not in st.session_state:
        st.session_state["markers"] = []
    if "map_data" not in st.session_state:
        st.session_state["map_data"] = {}
    if "all_drawings" not in st.session_state["map_data"]:
        st.session_state["map_data"]["all_drawings"] = None
    if "upload_file_button" not in st.session_state:
        st.session_state["upload_file_button"] = False


def reset_session_state():
    # Delete all the items in Session state besides center and zoom
    for key in st.session_state.keys():
        if key in ["center", "zoom"]:
            continue
        del st.session_state[key]
    initialize_session_state()


def initialize_map(center, zoom):
    m = folium.Map(location=center, zoom_start=zoom, scrollWheelZoom=False)
    draw = Draw(
        export=False,
        filename="custom_drawn_polygons.geojson",
        position="topright",
        draw_options={
            "polyline": False,  # disable polyline option
            "rectangle": False,  # disable rectangle option for now
            # enable polygon option
            #   'polygon': {'showArea': True, 'showLength': False, 'metric': False, 'feet': False},
            "polygon": False,  # disable rectangle option for now
            # enable circle option
            "circle": {
                "showArea": True,
                "showLength": False,
                "metric": False,
                "feet": False,
            },
            "circlemarker": False,  # disable circle marker option
            "marker": False,  # disable marker option
        },
        edit_options={"poly": {"allowIntersection": False}},
    )
    draw.add_to(m)
    MeasureControl(
        position="bottomleft",
        primary_length_unit="miles",
        secondary_length_unit="meters",
        primary_area_unit="sqmiles",
        secondary_area_unit=np.nan,
    ).add_to(m)
    return m


initialize_session_state()

m = initialize_map(center=st.session_state["center"], zoom=st.session_state["zoom"])

# Buttons
col1, col2, col3 = st.columns(3)

if col1.button("Add Pins"):
    st.session_state["markers"] = [
        folium.Marker(
            location=[random.randint(37, 38), random.randint(-97, -96)],
            popup="Test",
            icon=folium.Icon(icon="user", prefix="fa", color="lightgreen"),
        )
        for i in range(0, 10)
    ]

if col2.button("Clear Map", help="ℹ️ Click me to **clear the map and reset**"):
    reset_session_state()
    m = initialize_map(center=st.session_state["center"], zoom=st.session_state["zoom"])

with col3:
    st.markdown("##### Draw a circle by clicking the circle icon ----👇")

fg = folium.FeatureGroup(name="Markers")
for marker in st.session_state["markers"]:
    fg.add_child(marker)

# Create the map and store interaction data inside of session state
map_data = st_folium(
    m,
    center=st.session_state["center"],
    zoom=st.session_state["zoom"],
    feature_group_to_add=fg,
    key="new",
    width=1285,
    height=725,
    returned_objects=["all_drawings"],
    use_container_width=True,
)
st.write("## map_data")
st.write(map_data)
st.write("## session_state")
st.write(st.session_state)
