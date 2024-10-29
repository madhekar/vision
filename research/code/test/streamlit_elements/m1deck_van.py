import pydeck
import streamlit as st

DATA_URL = "https://raw.githubusercontent.com/visgl/deck.gl-data/master/examples/geojson/vancouver-blocks.json"
LAND_COVER = [
    [[-123.0, 49.196], [-123.0, 49.324], [-123.306, 49.324], [-123.306, 49.196]]
]

INITIAL_VIEW_STATE = pydeck.ViewState(
    latitude=49.254, longitude=-123.13, zoom=11, max_zoom=16, pitch=45, bearing=0
)

polygon = pydeck.Layer(
    "PolygonLayer",
    LAND_COVER,
    stroked=False,
    # processes the data as a flat longitude-latitude pair
    get_polygon="-",
    get_fill_color=[0, 0, 0, 20],
)

geojson = pydeck.Layer(
    "GeoJsonLayer",
    DATA_URL,
    opacity=0.8,
    stroked=False,
    filled=True,
    extruded=True,
    wireframe=True,
    get_elevation="properties.valuePerSqm / 20",
    get_fill_color="[255, 255, properties.growth * 255]",
    get_line_color=[255, 255, 255],
    pickable=True,
)

r = pydeck.Deck(layers=[polygon, geojson], initial_view_state=INITIAL_VIEW_STATE)

st.pydeck_chart(r)
r.to_html()
