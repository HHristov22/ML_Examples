"""
K-Means Clustering

Pros of K-Means Clustering:
- Simple, fast, and easy to implement.
- Scalable to large datasets.
- Works well when clusters have a spherical shape and are well-separated.
- Useful for partitioning large datasets into meaningful clusters.

Cons of K-Means Clustering:
- Requires the number of clusters (K) to be predefined, which may not be known in advance. - Elbow method
- Sensitive to the initial placement of centroids, which can lead to different results for different initializations.
- Struggles with complex shapes and non-spherical clusters.
- Prone to overfitting if the number of clusters is too high.
- Sensitive to outliers, which can distort the clusters.
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('./Clustering/K-Means/Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# ---------------------------------------------------------------------------------
# Finding the optimal number of clusters using KneeLocator
from kneed import KneeLocator

# Create a KneeLocator instance to find the elbow point
kneedle = KneeLocator(range(1, 11), wcss, curve='convex', direction='decreasing')
optimal_k = kneedle.knee

print('Optimal number of clusters:', optimal_k)
# ---------------------------------------------------------------------------------

# Training the K-Means model on the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()