import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import random
import string
import time

# https://www.toolify.ai/ai-model/xtuner-llava-llama-3-8b-v1-1-hf
def random_user_agent(num_chars=8):
    return "".join(random.choices(string.ascii_letters + string.digits, k=8))


cache = {}


# get location address information from latitude and longitude
def getLocationDetails(strLnL, max_retires):
    address = "na,na,na"

    if strLnL in cache:
        return cache[strLnL]
    else:
        geolocator = Nominatim(user_agent=random_user_agent())
        try:
            rev = RateLimiter(geolocator.reverse, min_delay_seconds=1)
            location = rev(strLnL, language="en", exactly_one=True)
            if location:
                address = location.address
                cache[strLnL] = address
                return address
        except (GeocoderTimedOut, GeocoderUnavailable) as e:
            print(f"Get address failed with {e}")
        print("->>>", address)
    return address

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

n = len(centroids)
colors = [plt.cm.plasma(i/n) for i in range(n)]
print(colors)
# Get addresses of the cluster centroids
# l = []
# for c in centroids:
#     print(c)
#     add = getLocationDetails((str(c[1]),str(c[0])),3)
#     l.append(add)
#     print(add)

# 4. Visualize the clustered data
plt.figure(figsize=(8, 6))
plt.scatter(x1, x2, c=labels, s=50, cmap='viridis') # Color points by cluster
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c=colors, label='Centroids') # Plot centroids
plt.title("K-Means Clustering Result")
plt.xlabel("lat")
plt.ylabel("lon")
plt.legend()
plt.grid(True)
plt.show()

# 5. Print the cluster centroids
print("Cluster Centroids:")
print(centroids)
