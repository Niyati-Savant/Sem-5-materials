import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(0)
df = pd.read_csv("Mumbai.csv")
df = df.tail(15)
prices = df['Price'].values
areas = df['Area'].values

data = np.vstack((prices, areas)).T

K = 2
# initial mean chosen randomly
random_indices = random.sample(range(len(data)), K)
print(f"Let random indices be {random_indices} i.e points--")
centroids = np.array([data[i] for i in random_indices])
print(centroids)

max_iters = 100
for _ in range(max_iters):
    # Assign each data point to the nearest centroid
    print(f"Distance Matrix for iteration {_}")
    distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
    print(distances)
    print(f"Clustering as per Minimum Distance")
    labels = np.argmin(distances, axis=0)
    print(labels)
    # Update the centroids based on the mean of points in each cluster
    new_centroids = np.array([data[labels == k].mean(axis=0) for k in range(K)])
    print("New Centeroids-")
    print(new_centroids)
    # Check for convergence
    if np.all(centroids == new_centroids):
        break

    centroids = new_centroids
    print("-----------------------------------------------")

# Visualize the clusters
colors = ['r', 'b']
for k in range(K):
    plt.scatter(data[labels == k][:, 0], data[labels == k][:, 1], c=colors[k], label=f'Cluster {k+1}')

plt.scatter(centroids[:, 0], centroids[:, 1], c='k', marker='x', label='Centroids')
plt.xlabel('Price')
plt.ylabel('Area')
plt.legend()
plt.show()