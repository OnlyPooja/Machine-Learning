import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons   # make_moons from sklearn.datasets to generate synthetic data for clustering.

# Generate synthetic data (you can replace this with your own data)
data, _ = make_moons(n_samples=200, noise=0.05, random_state=0)

# Perform DBSCAN clustering
dbscan = DBSCAN(eps=0.3, min_samples=5)
# eps=0.3The maximum distance between two samples for one to be considered as in the neighborhood of the other.
# min_samples=5: The minimum number of samples in a neighborhood for a data point to be considered a core point.
dbscan.fit(data)

# Visualize the clustered data
labels = dbscan.labels_
# A mask is created to identify core points by setting core_samples_mask to True for core points based on dbscan.core_sample_indices_
core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[dbscan.core_sample_indices_] = True

unique_labels = set(labels)
n_clusters = len(unique_labels) - (1 if -1 in labels else 0)

plt.figure(figsize=(8, 6))
# Colors are defined based on the number of unique labels to differentiate clusters.
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

for k, col in zip(unique_labels, colors):
    if k == -1:
        col = 'k'

    class_member_mask = (labels == k)

    xy = data[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)

    xy = data[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=6)

plt.title(f'DBSCAN Clustering (Estimated {n_clusters} Clusters)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
