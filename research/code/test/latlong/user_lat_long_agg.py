import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np


def get_num_of_clusters(X):

   # Range of K values to test
   range_n_clusters = range(2, 20)
   silhouette_scores = []

   for n_clusters in range_n_clusters:
        # Initialize KMeans with the current number of clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)

        # Calculate the silhouette score
        score = silhouette_score(X, cluster_labels)
        silhouette_scores.append(score)
        print(f"For n_clusters = {n_clusters}, silhouette score = {score:.4f}")
   return silhouette_scores.index(max(silhouette_scores))     

def get_cluster_centers(X, num_clusters):

    # Apply K-Means clustering
    # Initialize KMeans with the desired number of clusters (k=3 in this case)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10) # n_init for robustness
    kmeans.fit(X)

    # Get the cluster labels for each data point
    labels = kmeans.labels_

    # Get the coordinates of the cluster centroids
    centroids = kmeans.cluster_centers_

    return labels, centroids

if __name__=='__main__':
    df = pd.read_csv('lat_lon_nodup.csv')
    X = df.values.tolist()
    n_clusters = get_num_of_clusters(X)

    labels, centroids = get_cluster_centers(X, n_clusters + 2)

    print(centroids)
