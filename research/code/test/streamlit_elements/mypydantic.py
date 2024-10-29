import dataclasses
import json

import streamlit as st
from pydantic.json import pydantic_encoder

import streamlit_pydantic as sp


@dataclasses.dataclass
class LocationModel:
    name: str = 'unique location name key'
    desc: str = 'free form location description'
    lat: float = 0.0
    lon: float = 0.0


data = sp.pydantic_form(key="zesha_loc_form", model=LocationModel)
if data:
    st.write(dataclasses.asdict(data))
