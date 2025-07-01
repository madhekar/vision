import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from math import cos, asin, sqrt, radians, sin
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

"""
The "radius" parameter in NearestNeighbors algorithms, such as those found in scikit-learn, represents a
distance threshold. The units of this radius are dependent on the distance metric used. 
Essentially, the radius defines a "neighborhood" around each data point, and the algorithm identifies all other data points within that neighborhood. 
Here's how the radius unit is determined:

Inherited from the Distance Metric: The units of the radius are the same as the units of the chosen distance metric. For example, if you're using Euclidean distance and your data represents spatial coordinates in meters, then the radius will be in meters.
Euclidean Distance: This is the most common distance metric, measuring the straight-line distance between two points in a multi-dimensional space. If your data features are in units of meters, the radius for Euclidean distance will also be in meters.
Other Metrics: Different distance metrics exist, each with its own interpretation of distance. For instance:
    
    Manhattan Distance: Measures distance as the sum of absolute differences between coordinates.
    
    Cosine Similarity: Measures the angle between vectors, often used for text analysis.
    
    Haversine Distance: Used for calculating distances on a sphere, like the Earth. If you use this, the radius unit would be related to spherical units, 
    but you'd often convert to familiar units like miles or kilometers by multiplying by the Earth's radius. 

In summary, when using the "radius" parameter in NearestNeighbors, remember that its units are tied to the chosen distance metric and your data's feature units
"""
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

# ---
def haversine_dist(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in miles between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 3956  # Radius of earth in kilometers. Use 6371 for kilometers. Determines return value units.
    return c * r


def walk_centroids(X, cen, data_vs_clusters):
    ret = {}

    print(data_vs_clusters)

    # Initialize NearestNeighbors with your custom metric
    nbrs = NearestNeighbors( n_neighbors=data_vs_clusters, metric=haversine_distance, radius=10.0, leaf_size=30, p=2)

    # Fit the model
    nbrs.fit(X)

    for c in cen:
        distances, indices = nbrs.kneighbors([c.tolist()])
        ret[tuple(c.tolist())] = (indices)
    return ret


if __name__=='__main__':
    df = pd.read_csv('lat_lon_nodup.csv')

    X = df.values.tolist()
    len_X = len(X)

    n_clusters = get_num_of_clusters(X)

    labels, centroids = get_cluster_centers(X, n_clusters + 2)
    print(centroids)
    
    ret = walk_centroids(X, centroids,  10)#len_X// n_clusters)

    print(ret)
