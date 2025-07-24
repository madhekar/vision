import os
from sklearn.neighbors import BallTree
import numpy as np
import pandas as pd
import fastparquet as fp
#import geopandas as gpd


def read_parquet(p,f):
    df = None
    try:
        pf = fp.ParquetFile(os.path.join(p,f))
        df = pf.to_pandas()
        df['name'].str.strip()
    except Exception as e:
       print(f"error loading/ reading file: {e}")  
    return df

def create_balltree(df):
    df.dropna()
    print(df.describe())
    df.drop(columns=["name", "state", "country"], inplace=True)

    print(df.head(20)) 

    df = df[
            (df.longitude != "-")
            & (df.latitude != "-")
            & (~df.longitude.isnull())
            & (~df.latitude.isnull())
        ]  

    print(df.head(20))

    # df_lat = pd.to_numeric(df['latitude'], errors='coerce')
    # df_lon = pd.to_numeric(df['longitude'], errors='coerce')

    # print(df_lat)
    # print(df_lon)
   
    #df[["latitude", "longitude"]] = df[["latitude", "longitude"]].astype(float)

    #df = df.astype(float)

    cols = ['latitude','longitude']
    for col in cols:
        df[col] = pd.to_numeric(df[col] , errors='coerce')

    print(df.dtypes)
    print(df.head(20))

    np_data = df.to_numpy()
    
    print(np_data)

    np_data_radian = np.radians(np_data)

    print(np_data_radian)

    cleaned_array = np_data_radian[~np.isnan(np_data_radian).any(axis=1)]

    tree = BallTree(cleaned_array, metric="haversine")

def sample_ball_tree():
    # Sample data (replace with your actual data loading)
    np.random.seed(0)
    num_points = 1000000
    # Example: random points in a 10x10 area
    points = np.random.rand(num_points, 2) * 10

    print(points)

    # Convert to radians for BallTree (if using lat/lon)
    points_radians = np.radians(points)

    print(points_radians)

    # Build the BallTree (using haversine distance for lat/lon)
    tree = BallTree(points_radians, metric='haversine')

if __name__=='__main__':
    parquet_path = '/home/madhekar/work/home-media-app/data/app-data/static-metadata/locations/user-specific/Madhekar'
    parquet_file = 'static_locations.parquet'

    dff = read_parquet(parquet_path, parquet_file)

    create_balltree(dff)

    #sample_ball_tree()