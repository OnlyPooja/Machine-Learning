import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Generate random data for clustering
np.random.seed(0)
# This is the dataset you want to cluster.
data = np.random.rand(10, 2)

# Perform hierarchical clustering
# The 'single' linkage method defines the distance between two clusters (or data points) as the shortest distance between any two data points in the two clusters.
#  is performing hierarchical clustering on the data using the 'single' linkage method
linked = linkage(data, 'single')  # You can use different linkage methods like 'single', 'complete', 'average', etc.

# Create a dendrogram
plt.figure(figsize=(10, 5))
dendrogram(linked, orientation='top', labels=[f'Data point {i}' for i in range(len(data))], distance_sort='descending')
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data Points')
plt.ylabel('Distance')
plt.show()
