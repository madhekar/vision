import numpy as np
from scipy.spatial import cKDTree
import pandas as pd

EARTH_RADIUS_M = 6_371_000.0



def latlon_to_cartesian(latlon: np.ndarray) -> np.ndarray:
  rad = np.radians(latlon)
  lat, lon = rad[:, 0], rad[:, 1]
  x = np.cos(lat) * np.cos(lon)
  y = np.cos(lat) * np.sin(lon)
  z = np.sin(lat)
  return np.column_stack((x, y, z))


def meters_to_chord(meters: float) -> float:
  # Converts surface arc distance (meters) to 3D unit-sphere chord distance
  angle_rad = meters / EARTH_RADIUS_M
  return 2.0 * np.sin(angle_rad / 2.0)


def build_known_lat_long_tree(data_file):

    df = pd.read_csv(data_file)

    dfl = df[['latitude', 'longitude']]
    dfv = df[['description', 'state', 'country']]

    tree = cKDTree(latlon_to_cartesian(dfl.to_numpy())) 

    values = dfv.to_numpy()

    def query_nearest_neighbor_within(coordinates, distance=50_000):
       max_cord = meters_to_chord(distance)
       query_coorditates = np.array(coordinates)
       dist, idx = tree.query(latlon_to_cartesian(query_coorditates), k=1, distance_upper_bound=max_cord, workers=-1)

       for i, idx in enumerate(indices):
         if idx == len(coordinates):
            print(f"Query {i}: No match within radius")
         else:
            # Convert chord back to exact meters if needed
            arc_meters = 2.0 * EARTH_RADIUS_M * np.arcsin(distances[i] / 2.0)
            print(f"Query {i}: {values[idx]} ({arc_meters:.1f} m away)")


# # 1. Build the tree once
# coords = np.array([
#     [32.7157, -117.1611],  # San Diego
#     [34.0522, -118.2437],  # Los Angeles
#     [37.7749, -122.4194],  # San Francisco
# ])
# values = np.array(["San Diego", "Los Angeles", "San Francisco"])
# tree = cKDTree(latlon_to_cartesian(coords))

def query_nearest_neighbor_within(coordinates, distance=50_000):
  max_cord = meters_to_chord(distance)
  query_coorditates = np.array(coordinates)



# 2. Query nearest neighbor within 50 km (50,000 meters)
queries = np.array([
    [32.8000, -117.2000],  # ~10 km from San Diego (Match)
    [36.0000, -119.0000],  # ~230 km from nearest city (No match)
])

max_chord = meters_to_chord(50_000)

# workers=-1 uses all CPU cores for batch queries
distances, indices = tree.query(
    latlon_to_cartesian(queries),
    k=1,
    distance_upper_bound=max_chord,
    workers=-1,
)

# 3. Handle misses: out-of-range queries return index == len(coords) and dist == inf
for i, idx in enumerate(indices):
  if idx == len(coords):
    print(f"Query {i}: No match within radius")
  else:
    # Convert chord back to exact meters if needed
    arc_meters = 2.0 * EARTH_RADIUS_M * np.arcsin(distances[i] / 2.0)
    print(f"Query {i}: {values[idx]} ({arc_meters:.1f} m away)")
