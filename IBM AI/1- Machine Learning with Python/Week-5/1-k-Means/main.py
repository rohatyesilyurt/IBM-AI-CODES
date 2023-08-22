# k-Means on a randomly generated dataset
import random 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets import make_blobs 

# kendi verimizi oluşturalım
np.random.seed(0)

x, y = make_blobs(n_samples=5000, centers=[[4,4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)
plt.scatter(x[:, 0], x[:, 1], marker='.')

k_means = KMeans(init = "k-means++", n_clusters = 4, n_init = 12)
k_means.fit(x)


k_means_labels = k_means.labels_
k_means_cluster_centers = k_means.cluster_centers_
print(k_means_labels)
print(k_means_cluster_centers)