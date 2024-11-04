import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# Generate synthetic data
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)

# K-Means implementation
def kmeans(X, n_clusters, max_iters=100):
    centroids = X[np.random.choice(X.shape[0], n_clusters, replace=False)]
    for epoch in range(max_iters):
        # Assign clusters
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        # Compute new centroids
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(n_clusters)])
        
        # Check for convergence
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    return labels, centroids, epoch

# K-Means parameters
n_clusters = 3
initial_centroids = X[np.random.choice(X.shape[0], n_clusters, replace=False)]
labels_kmeans, final_centroids_kmeans, epoch_size_kmeans = kmeans(X, n_clusters)

# K-Means output
print("K-Means")
print("Initial Clusters:\n", initial_centroids)
print("Final Clusters:\n", final_centroids_kmeans)
print("Epoch Size:", epoch_size_kmeans)
print("Error Rate (Silhouette Score):", silhouette_score(X, labels_kmeans))

# Plotting K-Means results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=labels_kmeans, s=50, cmap='viridis')
plt.scatter(final_centroids_kmeans[:, 0], final_centroids_kmeans[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title('K-Means Clustering')

# DBSCAN implementation
def dbscan(X, eps=0.3, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    return labels

# DBSCAN parameters
eps = 0.3
min_samples = 5
labels_dbscan = dbscan(X, eps, min_samples)

# DBSCAN output
unique_labels = set(labels_dbscan)
n_clusters_dbscan = len(unique_labels) - (1 if -1 in unique_labels else 0)

print("\nDBSCAN")
print("Final Clusters (Labels):", labels_dbscan)
print("Number of Clusters (excluding noise):", n_clusters_dbscan)

# Plotting DBSCAN results
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=labels_dbscan, s=50, cmap='viridis')
plt.title('DBSCAN Clustering')

plt.tight_layout()
plt.show()
