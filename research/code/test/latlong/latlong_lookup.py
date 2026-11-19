import numpy as np
from scipy.spatial import cKDTree

# Sample data: (lat, lon) in degrees
coords = np.array([[32.7157, -117.1611], [34.0522, -118.2437], [37.7749, -122.4194]])
values = ["San Diego", "Los Angeles", "San Francisco"]


# Convert degrees to 3D unit sphere coordinates for accurate Euclidean distance
def latlon_to_cartesian(latlon):
  rad = np.radians(latlon)
  lat, lon = rad[:, 0], rad[:, 1]
  x = np.cos(lat) * np.cos(lon)
  y = np.cos(lat) * np.sin(lon)
  z = np.sin(lat)
  return np.column_stack((x, y, z))


tree = cKDTree(latlon_to_cartesian(coords))

# Query nearest point in microseconds
query_pt = latlon_to_cartesian(np.array([[32.72, -117.16]]))
dist, idx = tree.query(query_pt, k=1)
print(values[idx[0]])  # "San Diego"
