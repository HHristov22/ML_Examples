"""
Hierarchical Clustering

Pros of Hierarchical Clustering:
- No need to predefine the number of clusters, as the dendrogram provides a visual way to decide the optimal number.
- Can capture complex cluster relationships and hierarchical structures.
- Suitable for smaller datasets and provides an intuitive visual representation of the data via dendrograms.
- Works well with different types of distance metrics (e.g., Euclidean, Manhattan, etc.).

Cons of Hierarchical Clustering:
- Computationally expensive for large datasets, as the algorithm has a complexity of O(n^3), making it slower than other clustering methods.
- Sensitive to noise and outliers, which can distort the clustering structure.
- Once a cluster is formed, it cannot be undone (no "re-clustering").
- Less efficient and scalable compared to algorithms like K-Means for very large datasets.
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('./Clustering/Hierachical/Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

# Training the Hierarchical Clustering model on the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, linkage = 'ward')
y_hc = hc.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')

plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()