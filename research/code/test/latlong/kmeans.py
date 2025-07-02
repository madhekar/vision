import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import pandas as pd
import numpy as np

# 1. Generate sample data
# Create a synthetic dataset with 3 distinct clusters for demonstration
#X,y = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)


# df = pd.read_csv("lat_lon_nodup.csv")
# X = df.values.tolist()

df = pd.read_csv(
    "/home/madhekar/work/home-media-app/data/input-data-1/error/img/missing-data/AnjaliBackup/missing-metadata-wip.csv"
)  #'lat_lon_nodup.csv')

df = df.drop(["SourceFile", "DateTimeOriginal"], axis=1)

dfm = df[~(df["GPSLatitude"] == "-")]

dfm[["GPSLatitude", "GPSLongitude"]] = (
    dfm[["GPSLatitude", "GPSLongitude"]].apply(pd.to_numeric).round(6)
)

print(dfm.head())
X = dfm.values.tolist()


x1 = np.array(X)[:,0]
x2 = np.array(X)[:,1]
# 2. Visualize the original data (optional)
plt.figure(figsize=(18, 16))
plt.scatter(x1, x2,s=50)
plt.title("Original Data Points")
plt.xlabel("lat")
plt.ylabel("lon")
plt.grid(True)
plt.show()

# 3. Apply K-Means clustering
# Initialize KMeans with the desired number of clusters (k=3 in this case)
kmeans = KMeans(n_clusters=20, random_state=0, n_init=10) # n_init for robustness
kmeans.fit(X)

# Get the cluster labels for each data point
labels = kmeans.labels_

# Get the coordinates of the cluster centroids
centroids = kmeans.cluster_centers_
print(centroids)

# 4. Visualize the clustered data
plt.figure(figsize=(8, 6))
plt.scatter(x1, x2, c=labels, s=50, cmap='viridis') # Color points by cluster
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='red', label='Centroids') # Plot centroids
plt.title("K-Means Clustering Result")
plt.xlabel("lat")
plt.ylabel("lon")
plt.legend()
plt.grid(True)
plt.show()

# 5. Print the cluster centroids
print("Cluster Centroids:")
print(centroids)
