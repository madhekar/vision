from vptree import VPTree  # Assuming a library like the one by RickardSjogren

# 1. Define a distance function (e.g., Euclidean distance for 2D points)
def euclidean_distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

# Sample data (list of points/hashes)
data_points = [(1, 2), (1, 2), (3, 4), (5, 6), (3, 4), (7, 8)]

# 2. Build the VP-tree
tree = VPTree(data_points, euclidean_distance)

# 3. Find duplicates:
# Iterate through each point and find neighbors within a distance of 0 (exact match)
duplicates = set()
for point in data_points:
    # Query the tree for neighbors within a distance of 0
    # The result usually returns a list of (distance, item) tuples
    neighbors = tree.get_n_nearest_neighbors(point, 1)
    
    # If more than one result is returned, it means a duplicate exists
    if len(neighbors) > 1:
        # Add the point to the set of duplicates (sets handle unique duplicates)
        duplicates.add(point)

print("Duplicate points found:", list(duplicates))
