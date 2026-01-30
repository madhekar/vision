from sklearn.neighbors import BallTree
import numpy as np
import pandas as pd

class GeoBallTree():
    def __init__(self, df):
        self.df = df
        self.df_copy = self.df.copy(deep=True)
        self.np_arr_rad_clean = None
        self.BTree = None

    def create_data_structure(self):
      
       self.df.dropna()

       self.df.drop(columns=['name', "state", "country"], inplace=True)

       #print(df.head(20)) 

       self.df = self.df[(self.df.longitude != "-") & (self.df.latitude != "-") & (~self.df.longitude.isnull()) & (~self.df.latitude.isnull())] 
       
       cols = ['latitude','longitude']
       for col in cols:
           self.df[col] = pd.to_numeric(self.df[col] , errors='coerce')

       #print(self.df.head(20))

       np_data = self.df.to_numpy()
    
       #print(np_data)

       np_clean_array = np_data[~np.isnan(np_data).any(axis=1)]

       self.np_arr_rad_clean = np.deg2rad(np_clean_array)

       #print(clean_array)

       self.BTree = BallTree(self.np_arr_rad_clean, leaf_size=15, metric="haversine")



    def find_nearest(self, lat, lon):

        arr = np.array([[lat, lon]])

        # print(f'search nearest point to: {arr}')

        query_pt_radians = np.deg2rad(arr)

        # print(f'search nearest point to (rad) :{query_pt_radians}')

        dist, index = self.BTree.query(query_pt_radians, k=1)

        # print(f'dist: {dist[0][0]} index: {index} ind: {index[0]}')

        nearest_pt_rad = self.np_arr_rad_clean[index[0]]

        nearest_pt = np.rad2deg(nearest_pt_rad)

        if dist >= 2.0:
            return None, dist
        else:
            return nearest_pt, dist
        
    def find_nearest_location_name(self, np_arr):
    
        if np_arr is None:
            return "none"
        else:
                npa = np_arr[0]

                q = self.df_copy[(np.isclose(self.df_copy["latitude"] ,npa[0])) & (np.isclose(self.df_copy["longitude"], npa[1]))]

        if q.empty:
            ret = 'none'
        else:
            ret = q['name'].item() 

        return ret           
    
    def get_location_name_for_latlong(self, lat, lon):

        if (( lat is  None) & (lon is None)):
            return ""

        npt, d = self.find_nearest(lat, lon)

        nm = self.find_nearest_location_name(npt)

        return nm

