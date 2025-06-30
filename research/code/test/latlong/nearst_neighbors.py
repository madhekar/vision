from sklearn.neighbors import NearestNeighbors
from math import cos, asin, sqrt
import numpy as np
import pandas as pd

# Your custom distance metric (e.g., Manhattan distance)
def manhattan_distance(x, y):
  return np.sum(np.abs(x - y))

"""
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295
    hav = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p)*cos(lat2*p) * (1-cos((lon2-lon1)*p)) / 2
    return 12742 * asin(sqrt(hav))
"""

def haversine_distance(lat_lon1, lat_lon2):
    p = 0.017453292519943295
    hav = (
        0.5
        - cos((lat_lon2[0] - lat_lon1[0]) * p) / 2
        + cos(lat_lon1[0] * p) * cos(lat_lon2[0] * p) * (1 - cos((lat_lon2[1] - lat_lon1[1]) * p)) / 2
    )
    return 12742 * asin(sqrt(hav))


def walk_centroids(X, cen):

  # Initialize NearestNeighbors with your custom metric
  nbrs = NearestNeighbors(n_neighbors=20, metric=haversine_distance, radius=5.0, leaf_size=30, p=2)

  # Fit the model
  nbrs.fit(X)

  # Find nearest neighbors for a new point
  #new_point = np.array([[117.28210047 , 33.010097  ]])

  for c in cen:
     distances, indices = nbrs.kneighbors(c)
     print("Distances:", distances)
     print("Indices:", indices)  


df = pd.read_csv("lat_lon_nodup.csv")
X = df.values.tolist()
