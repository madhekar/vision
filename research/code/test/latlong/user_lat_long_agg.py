import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from math import cos, asin, sqrt
import numpy as np


def get_num_of_clusters(X):

   # Range of K values to test
   range_n_clusters = range(2, 20)
   silhouette_scores = []

   for n_clusters in range_n_clusters:
        # Initialize KMeans with the current number of clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(X)

        # Calculate the silhouette score
        score = silhouette_score(X, cluster_labels)
        silhouette_scores.append(score)
        print(f"For n_clusters = {n_clusters}, silhouette score = {score:.4f}")
   return silhouette_scores.index(max(silhouette_scores))     

def get_cluster_centers(X, num_clusters):

    # Apply K-Means clustering
    # Initialize KMeans with the desired number of clusters (k=3 in this case)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto') # n_init for robustness
    kmeans.fit(X)

    # Get the cluster labels for each data point
    labels = kmeans.labels_

    # Get the coordinates of the cluster centroids
    centroids = kmeans.cluster_centers_

    return labels, centroids

def haversine_distance(lat_lon1, lat_lon2):
    p = 0.017453292519943295
    hav = (
        0.5
        - cos((lat_lon2[0] - lat_lon1[0]) * p) / 2
        + cos(lat_lon1[0] * p)
        * cos(lat_lon2[0] * p)
        * (1 - cos((lat_lon2[1] - lat_lon1[1]) * p))
        / 2
    )
    return 12742 * asin(sqrt(hav))


def walk_centroids(X, cen):
    ret = {}

    # Initialize NearestNeighbors with your custom metric
    nbrs = NearestNeighbors( n_neighbors=20, metric=haversine_distance, radius=5.0, leaf_size=30, p=2)

    # Fit the model
    nbrs.fit(X)

    # Find nearest neighbors for a new point
    # new_point = np.array([[117.28210047 , 33.010097  ]])

    for c in cen:
        print(c.tolist())
        distances, indices = nbrs.kneighbors([c.tolist()])
        ret[tuple(c.tolist())] = (indices)
    return ret


if __name__=='__main__':
    df = pd.read_csv('lat_lon_nodup.csv')
    X = df.values.tolist()
    n_clusters = get_num_of_clusters(X)

    labels, centroids = get_cluster_centers(X, n_clusters + 2)
    print(centroids)
    
    ret = walk_centroids(X, centroids)

    print(ret)
