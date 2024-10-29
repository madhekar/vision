import os
import dataclasses
import numpy as np
import pandas as pd
#import json

import streamlit as st
#from pydantic.json import pydantic_encoder

import streamlit_pydantic as sp


static_metadata_path= '/home/madhekar/work/home-media-app/data/app-data/static-metadata/'
static_metadata_file= 'static-metadata.csv'



df1 = (pd.read_csv(os.path.join(static_metadata_path, static_metadata_file))) 
print(df1.columns)
#df1.columns(['name', 'desc', 'lat', 'lon'])
my_table = st.table(df1)

print(my_table)
@dataclasses.dataclass
class LocationModel:
    name: str = 'unique location name key'
    nesc: str = 'free form location description'
    lat: float = 0.0
    lon: float = 0.0


data = sp.pydantic_form(key="zesha_loc_form", model=LocationModel)
if data:
    st.write(dataclasses.asdict(data))
    st.write(pd.DataFrame.from_dict(dataclasses.asdict(data), orient="index"))
    my_table.add_rows(pd.DataFrame.from_dict(dataclasses.asdict(data), orient='index').transpose())