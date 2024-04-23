# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import necessary libraries: pandas, numpy, matplotlib.pyplot, and specific modules from sklearn.
2.Read the data from the CSV file into a Pandas DataFrame.
3.Extract the relevant features ('Annual Income (k$)' and 'Spending Score (1-100)') into a variable X.
4.Visualize the data points using matplotlib.pyplot.scatter.
5.Define the number of clusters (k) as 5 and initialize a K-means clustering model.
6.Fit the model to the data and obtain cluster centroids and labels.
7.Visualize the clustered data points, centroids, and radii using matplotlib.


## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: K Shakthi Sundar
RegisterNumber:  212222040152
*/
```
```

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
data=pd.read_csv('/content/Mall_Customers_EX8.csv')
data
X=data[['Annual Income (k$)', 'Spending Score (1-100)']]
X
plt.figure(figsize=(4, 4))
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()
k=5
kmeans=KMeans(n_clusters=k)
kmeans.fit(X)
centroids=kmeans.cluster_centers_
labels=kmeans.labels_
print("Centroids:")
print(centroids)
print("Labels:")
print(labels)
colors = ['r', 'g', 'b','c', 'm']
for i in range(k):
    cluster_points = X[labels == i]
    plt.scatter(cluster_points['Annual Income (k$)'], cluster_points['Spending Score (1-100)'], 
           color=colors[i], label=f'Cluster {1+1}')
    distances = euclidean_distances(cluster_points, [centroids[i]])
    radius = np.max(distances)
    circle = plt.Circle(centroids[i], radius, color=colors[i], fill=False)
    plt.gca().add_patch(circle)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, color='k', label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.axis('equal') # Ensure aspect ratio is equal
plt.show()
```

## Output:
### Data Set:
![image](https://github.com/DhanushPalani/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/121594640/c36b634b-54a8-45d4-b998-845adaa00f30)
### Data:
![image](https://github.com/DhanushPalani/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/121594640/59a35cb9-79c3-4218-b7f6-cb0180aabf2e)
### Centroid:
![image](https://github.com/ShakthiSundar-K/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/128116143/b47b8130-b4c7-4b09-87f3-5b7de224c288)
### K-Means Clustering:
![image](https://github.com/DhanushPalani/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/121594640/8fe1a610-cb8c-49c0-9914-263a972a6276)



## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
