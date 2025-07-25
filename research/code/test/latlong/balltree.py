import os
from sklearn.neighbors import BallTree
import numpy as np
import pandas as pd
import fastparquet as fp
#import geopandas as gpd

# def conditional_proc(x):
#     if isinstance(x, float):
#         return np.radians(x)
#     else:
#         return x
    
def read_parquet(p,f):
    df = None
    try:
        pf = fp.ParquetFile(os.path.join(p,f))
        df = pf.to_pandas()
        df['name'] = df['name'].str.strip()
    except Exception as e:
       print(f"error loading/ reading file: {e}")  
    return df

def build_balltree(df):

    df.dropna()

    df.drop(columns=['name', "state", "country"], inplace=True)

    #print(df.head(20)) 

    df = df[(df.longitude != "-") & (df.latitude != "-") & (~df.longitude.isnull()) & (~df.latitude.isnull())]  

    #print(df.head(20))

    cols = ['latitude','longitude']
    for col in cols:
        df[col] = pd.to_numeric(df[col] , errors='coerce')

    #print(df.head(20))

    np_data = df.to_numpy()
    
    #print(np_data)

    clean_array = np_data[~np.isnan(np_data).any(axis=1)]

    clean_array_rad = np.deg2rad(clean_array)

    #print(clean_array)

    tree = BallTree(clean_array_rad,  metric="haversine")

    return tree, clean_array_rad

def find_nearest(bt, np_arr_rad, lat, lon):
        
        arr = np.array([[lat,lon]])

        #print(f'search nearest point to: {arr}')

        query_pt_radians = np.deg2rad(arr)

        #print(f'search nearest point to (rad) :{query_pt_radians}')

        dist, index = bt.query(query_pt_radians, k=1)

        #print(f'dist: {dist[0][0]} index: {index} ind: {index[0]}')

        nearest_pt_rad = np_arr_rad[index[0]]

        nearest_pt = np.rad2deg(nearest_pt_rad)

        if dist >= 2.0:
            return None, dist
        else: 
            return nearest_pt, dist
        

def find_nearest_location_name(df, np_arr):
    
    if np_arr is None:
       return None
    else:
        npa = np_arr[0]

        q = df[(df["latitude"] == str(npa[0])) & (df["longitude"] == str(npa[1]))]

    return q['name'].item()


if __name__=='__main__':
    parquet_path = '/home/madhekar/work/home-media-app/data/app-data/static-metadata/locations/user-specific/Madhekar'
    parquet_file = 'static_locations.parquet'

    lat = 37.809326
    lon = -122.409981
    dff = read_parquet(parquet_path, parquet_file)
    df_copy = dff.copy(deep=True)
    #print(f' original: {df_copy.head()}')

    BT,arr_rad = build_balltree(dff)

    nept, d = find_nearest(BT, arr_rad, lat, lon)

    loc_name = find_nearest_location_name(df_copy, nept)

    print(f'nearest point to {lat} : {lon} is: {nept} and distance: {d}  location name: {loc_name}')

