from sklearn.neighbors import NearestNeighbors
import numpy as np

# Your custom distance metric (e.g., Manhattan distance)
def manhattan_distance(x, y):
  return np.sum(np.abs(x - y))

# Example data
X = np.array([[1, 2], [3, 4], [5, 6]])

# Initialize NearestNeighbors with your custom metric
nbrs = NearestNeighbors(n_neighbors=2, metric=manhattan_distance)

# Fit the model
nbrs.fit(X)

# Find nearest neighbors for a new point
new_point = np.array([[2, 3]])
distances, indices = nbrs.kneighbors(new_point)

print("Distances:", distances)
print("Indices:", indices)
