import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('/home/madhekar/work/home-media-app/data/input-data-1/error/img/missing-data/AnjaliBackup/missing-metadata-wip.csv')#'lat_lon_nodup.csv')

df =df.drop(['SourceFile', 'DateTimeOriginal'], axis=1)

dfm = df[~(df['GPSLatitude'] == '-')]  

dfm[["GPSLatitude", "GPSLongitude"]] = dfm[["GPSLatitude", "GPSLongitude"]].apply(pd.to_numeric).round(6)

print(dfm.head())
X = dfm.values.tolist()

print(X)
# Range of K values to test
range_n_clusters = range(2, 50)

silhouette_scores = []

for n_clusters in range_n_clusters:
    # Initialize KMeans with the current number of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)

    # Calculate the silhouette score
    score = silhouette_score(X, cluster_labels)
    silhouette_scores.append(score)
    print(f"For n_clusters = {n_clusters}, silhouette score = {score:.4f}")

# Plotting the silhouette scores
plt.figure(figsize=(8, 6))
plt.plot(range_n_clusters, silhouette_scores, marker="o")
plt.title("Silhouette Score for Various K Values")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Average Silhouette Score")
plt.grid(True)
plt.show()