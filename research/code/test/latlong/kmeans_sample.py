# import neccessaries librariesimport pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples,silhouette_score
# load petal data
data = datasets.load_iris() #dir(data)
# load into Dataframe 
df = pd.DataFrame(data.data,columns = data.feature_names)
print(df.shape)
print(df.head())
df1 = df.drop(['sepal length (cm)', 'sepal width (cm)'],axis = 'columns')
print(df1.head())
# plot scatter plot
plt.scatter(df1['petal length (cm)'],df1['petal width (cm)'])
# Now check silhouette coefficient
for i,k in enumerate([2,3,4,5]):
    
    fig, ax = plt.subplots(1,2,figsize=(15,5))
    
    # Run the kmeans algorithm
    km = KMeans(n_clusters=k)
    y_predict = km.fit_predict(df1)
    centroids  = km.cluster_centers_
# get silhouette
silhouette_vals = silhouette_samples(df1,y_predict)
    #silhouette_vals
# silhouette plot
y_ticks = []
y_lower = y_upper = 0
for i,cluster in enumerate(np.unique(y_predict)):
   cluster_silhouette_vals = silhouette_vals[y_predict ==cluster]
   cluster_silhouette_vals.sort()
   y_upper += len(cluster_silhouette_vals)
   
   ax[0].barh(range(y_lower,y_upper),
   cluster_silhouette_vals,height =1);
   ax[0].text(-0.03,(y_lower+y_upper)/2,str(i+1))
   y_lower += len(cluster_silhouette_vals)
       
   # Get the average silhouette score 
   avg_score = np.mean(silhouette_vals)
   ax[0].axvline(avg_score,linestyle ='--',
   linewidth =2,color = 'green')
   ax[0].set_yticks([])
   ax[0].set_xlim([-0.1, 1])
   ax[0].set_xlabel('Silhouette coefficient values')
   ax[0].set_ylabel('Cluster labels')
   ax[0].set_title('Silhouette plot for the various clusters');
    
    
   # scatter plot of data colored with labels
    
   ax[1].scatter(df['petal length (cm)'],
   df['petal width (cm)'] , c = y_predict);
   ax[1].scatter(centroids[:,0],centroids[:,1],marker = '*' , c= 'r',s =250);
   ax[1].set_xlabel('Eruption time in mins')
   ax[1].set_ylabel('Waiting time to next eruption')
   ax[1].set_title('Visualization of clustered data', y=1.02)
    
   plt.tight_layout()
   plt.title(f' Silhouette analysis using k = {k}',fontsize=16,fontweight = 'semibold')
   plt.savefig(f'Silhouette_analysis_{k}.jpg')