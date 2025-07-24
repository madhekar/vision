from sklearn.neighbors import BallTree
import numpy as np

class GeoBallTree():
    def __init__(self, df):
        self.df = df
        self.lpt = None
        self.lrad = None
        self.BT = None

    def create_data_structure(self):
        self.lpt = self.df[['latitude','longitude']].astype(float).to_numpy()    
        print(self.lpt)
        self.lrad = np.radians(self.lpt)
        self.BT = BallTree(self.lrad, metric='haversine')

    def query_find_nearest(self, lat, lon):
       
       query_pt_radians = np.radians([lat, lon])

       dist, index = self.BT.query(query_pt_radians, k=1)

       nearest_pt_rad = self.lrad(index[0])

       nearest_pt = np.degrees(nearest_pt_rad)

       return nearest_pt, dist