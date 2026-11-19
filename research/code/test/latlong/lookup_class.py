import numpy as np
from scipy.spatial import cKDTree
import pandas as pd

EARTH_RADIUS_M = 6_371_000.0


class SpatialMatcher:

  def __init__(self):
    self.tree = None
    self.values = None

  def init_tree(self, data_file):
    column_names = ['description', 'state', 'country', 'latitude', 'longitude']
    df = pd.read_csv(data_file, names=column_names, header=None)

    coords = df[['latitude', 'longitude']].to_numpy(dtype=float)
    self.values = df[['description', 'state', 'country']].to_numpy()
    self.tree = cKDTree(self.latlon_to_cartesian(coords))

  def query(self, target_geo_coord, max_chord_val=50_000):
    if self.tree is None:
      raise RuntimeError('Tree not initialized. Call init_tree() first.')

    max_chord = self.meters_to_chord(max_chord_val)
    query_coordinates = np.atleast_2d(np.asarray(target_geo_coord, dtype=float))

    distances, indices = self.tree.query(
        self.latlon_to_cartesian(query_coordinates),
        k=1,
        distance_upper_bound=max_chord,
        workers=-1,
    )

    for i, idx in enumerate(indices):
      if idx >= len(self.values):
        print(f'Query {i}: No match within {max_chord_val} m')
      else:
        arc_meters = 2.0 * EARTH_RADIUS_M * np.arcsin(distances[i] / 2.0)
        print(f'Query {target_geo_coord}: {self.values[idx]} ({arc_meters:.1f} m away)')

  @staticmethod
  def latlon_to_cartesian(latlon: np.ndarray) -> np.ndarray:
    rad = np.radians(latlon)
    lat, lon = rad[:, 0], rad[:, 1]
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.column_stack((x, y, z))

  @staticmethod
  def meters_to_chord(meters: float) -> float:
    angle_rad = meters / EARTH_RADIUS_M
    return 2.0 * np.sin(angle_rad / 2.0)


sm = SpatialMatcher()
sm.init_tree(
    '/mnt/zmdata/home-media-app/data/app-data/static-metadata/locations/default/default.csv'
)
sm.query([32.8456, -117.2734])
# Query [32.8456, -117.2734]: ['La Jolla Cave La Jolla' 'CA' 'US'] (394.6 m away)
