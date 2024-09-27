# K-Means vs Hierarchical Clustering

| Feature                  | K-Means Clustering                      | Hierarchical Clustering                  |
|--------------------------|-----------------------------------------|------------------------------------------|
| **Algorithm Type**       | Partitioning                            | Hierarchical (Agglomerative or Divisive) |
| **Number of Clusters**   | Must specify **K** beforehand           | Decided by interpreting the dendrogram   |
| **Cluster Shape**        | Assumes spherical clusters              | Can find clusters of various shapes      |
| **Speed**                | Fast, efficient for large datasets      | Slower, not ideal for large datasets     |
| **Memory Usage**         | Low                                      | High (needs distance matrix)             |
| **Determinism**          | Non-deterministic (depends on initialization) | Deterministic                      |
| **Result Interpretation**| Flat clusters                           | Hierarchical relationships (dendrogram)  |
| **Main Advantage**       | Simple and easy to implement            | No need to pre-specify number of clusters|
| **Main Disadvantage**    | Need to choose **K**                    | Computationally intensive                |
