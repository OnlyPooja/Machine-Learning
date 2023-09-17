import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering

# Generate synthetic data (you can replace this with your own data)
data, _ = make_blobs(n_samples=300, centers=4, random_state=0, cluster_std=1.0)

# Perform agglomerative hierarchical clustering
agg_clustering = AgglomerativeClustering(n_clusters=4)
agg_clustering.fit(data)

# Visualize the clustered data
plt.scatter(data[:, 0], data[:, 1], c=agg_clustering.labels_, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Agglomerative Clustering')
plt.show()
