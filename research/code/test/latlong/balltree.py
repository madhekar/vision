import os
from sklearn.neighbors import BallTree
import numpy as np
import pandas as pd
import fastparquet as fp
import glob
from GPSPhoto import gpsphoto

import time
#import geopandas as gpd

# def conditional_proc(x):
#     if isinstance(x, float):
#         return np.radians(x)
#     else:
#         return x


    
def read_parquet(p,f):
    df = None
    try:
        #pf = fp.ParquetFile(os.path.join(p,f))     
        df = pd.read_parquet(os.path.join(p,f), 'fastparquet')
        # print(pf.schema, ':', pf.dtypes)
        # df = pf.to_pandas()
        
        # df['name'] = df.name.astype(str)
        # df['state'] = df.state.astype(str)
        # df['country'] = df.country.astype(str)
        # df["latitude"] = df.latitude.astype(float)
        # df["longitude"] = df.longitude.astype(float)

        df['name'] = df['name'].str.strip()
        print('**', df.dtypes)
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

    tree = BallTree(clean_array_rad, leaf_size=15, metric="haversine")

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

        print(f'nearest pt: {nearest_pt}')

        if dist >= 10.0:
            return None, dist
        else: 
            return nearest_pt, dist
        
import math
def find_nearest_location_name(df, np_arr):
    
    if np_arr is None:
       return None
    else:
        npa = np_arr[0]

        q = df[(math.isclose(float(df['latitude']), npa[0])) & (math.isclose(float(df["longitude"]), npa[1]))]

        print(f'{np_arr} --> {q}')

        if q.empty:
            ret = 'none'
        else:
            ret = q['name'].item() 

    return ret

def gpsInfo(img):
    gps = ()
    try:
        # Get the data from image file and return a dictionary
        data = gpsphoto.getGPSData(img)

        if "Latitude" in data and "Longitude" in data:
            gps = (round(data["Latitude"], 6), round(data["Longitude"], 6))
    except Exception as e:
        print(f"exception occurred in extracting lat/ lon data: {e}")
    return gps


def getRecursive(rootDir, chunk_size=10):
    f_list = []
    for fn in glob.iglob(rootDir + "/**/*", recursive=True):
        if not os.path.isdir(os.path.abspath(fn)):
            f_list.append(os.path.abspath(fn))
    for i in range(0, len(f_list), chunk_size):
        yield f_list[i : i + chunk_size]

if __name__=='__main__':
    images_path = '/home/madhekar/work/home-media-app/data/input-data-1/img/Madhekar'
    parquet_path = '/home/madhekar/work/home-media-app/data/app-data/static-metadata/locations/user-specific/Madhekar'
    parquet_file = 'static_locations.parquet'

    lat = 37.809326
    lon = -122.409981
    dff = read_parquet(parquet_path, parquet_file)
    df_copy = dff.copy(deep=True)
    #print(f' original: {df_copy.head()}')

    BT,arr_rad = build_balltree(dff)

    # get path
    # get url
    # get

    img_iterator = getRecursive(images_path, chunk_size=1)

    for i in img_iterator:

       ll = gpsInfo(i[0])
       if ll != ():
        print(i, ':', ll)

        nept, d = find_nearest(BT, arr_rad, ll[0], ll[1])

        loc_name = find_nearest_location_name(df_copy, nept)

        print(f'nearest point to {lat} : {lon} is: {nept} and distance: {d}  location name: {loc_name}')

        time.sleep(2)

