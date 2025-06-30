import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 1. Generate sample data
# Create a synthetic dataset with 3 distinct clusters for demonstration
X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)

# 2. Visualize the original data (optional)
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.title("Original Data Points")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()

# 3. Apply K-Means clustering
# Initialize KMeans with the desired number of clusters (k=3 in this case)
kmeans = KMeans(n_clusters=3, random_state=0, n_init=10) # n_init for robustness
kmeans.fit(X)

# Get the cluster labels for each data point
labels = kmeans.labels_

# Get the coordinates of the cluster centroids
centroids = kmeans.cluster_centers_

# 4. Visualize the clustered data
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis') # Color points by cluster
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='red', label='Centroids') # Plot centroids
plt.title("K-Means Clustering Result")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()

# 5. Print the cluster centroids
print("Cluster Centroids:")
print(centroids)
