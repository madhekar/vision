from sklearn.neighbors import BallTree
import numpy as np
import geopandas as gpd

# Sample data (replace with your actual data loading)
np.random.seed(0)
num_points = 1000000
# Example: random points in a 10x10 area
points = np.random.rand(num_points, 2) * 10

# Convert to radians for BallTree (if using lat/lon)
points_radians = np.radians(points)

# Build the BallTree (using haversine distance for lat/lon)
tree = BallTree(points_radians, metric='haversine')