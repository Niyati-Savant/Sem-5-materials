import pandas as pd
import scipy.cluster.hierarchy as sch
import numpy as np
import matplotlib.pyplot as plt

# Load your house price dataset
df = pd.read_csv("Agglomeritive.csv")
data = df.head(5)

# Select the columns you want to use for clustering
columns_of_interest = ['Price', 'Area']

X = df[columns_of_interest].values
print(X)

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


# Single linkage clustering
def single_linkage(X):
    num_samples = X.shape[0]
    linkage_matrix = np.zeros((num_samples - 1, 4))

    clusters = [[i] for i in range(num_samples)]

    for i in range(num_samples - 1):
        min_distance = float('inf')
        merge_candidates = (0, 0)

        for cluster1_idx in range(len(clusters)):
            for cluster2_idx in range(cluster1_idx + 1, len(clusters)):
                for sample1_idx in clusters[cluster1_idx]:
                    for sample2_idx in clusters[cluster2_idx]:
                        distance = euclidean_distance(X[sample1_idx], X[sample2_idx])
                        if distance < min_distance:
                            min_distance = distance
                            merge_candidates = (cluster1_idx, cluster2_idx)

        merged_cluster = clusters.pop(merge_candidates[1])
        clusters[merge_candidates[0]].extend(merged_cluster)

        linkage_matrix[i, 0] = merge_candidates[0]
        linkage_matrix[i, 1] = merge_candidates[1]
        linkage_matrix[i, 2] = min_distance
        linkage_matrix[i, 3] = len(merged_cluster)

    return linkage_matrix


linkage_matrix_1 = single_linkage(X)

# Dendrogram
dendrogram = sch.dendrogram(linkage_matrix_1, labels=np.arange(X.shape[0]))
plt.title('Dendrogram')
plt.xlabel('Data Points')
plt.ylabel('Euclidean Distances')
plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
#
# # Data points
# data = np.array([[0.40, 0.53],
#                  [0.22, 0.38],
#                  [0.35, 0.32],
#                  [0.26, 0.19],
#                  [0.08, 0.41],
#                  [0.45, 0.30]])
#
# # Create clusters for each data point
# clusters = [{i} for i in range(len(data))]
#
# # Calculate Euclidean distance between two points
# def euclidean_distance(point1, point2):
#     X = np.sqrt(np.sum((point1 - point2) ** 2))
#     return round(X,2)
#
# # Create a distance matrix to store pairwise distances
# distance_matrix = np.zeros((len(data), len(data)))
#
# for i in range(len(data)):
#     for j in range(len(data)):
#
#         distance = euclidean_distance(data[i], data[j])
#         distance_matrix[i][j] = distance
#
# for i in range(len(data)):
#     for j in range(len(data)):
#         if (j<=i):
#             print(distance_matrix[i][j],end='\t')
#     print('\t')
#
# # Assuming 'distance_matrix' is your matrix
# min_value = 100
#
# # Iterate through the matrix to find the minimum value
# for i in range(len(data)):
#     for j in range(len(data)):
#         if i !=j:
#             if distance_matrix[i][j] < min_value :
#                 min_value = distance_matrix[i][j]
#                 m1=i
#                 m2=j
#
# print("The minimum value in the matrix is:", min_value)
# print(f"Merge {m1} and {m2}")
#
# # Combine points m1 and m2 and update the clusters list # Remove the cluster at index m2
# merged_cluster = clusters[m1].union(clusters[m2])
# clusters[m1] = merged_cluster
# clusters.pop(m2)
# print(clusters)
#
# # Update the distance_matrix
# for i in range(len(data)):
#     if i == m1:
#         continue
#     if i != m2:
#         # Update the distance to the merged cluster using the minimum distance
#         distance_matrix[m1][i] = min(distance_matrix[m1][i], distance_matrix[m2][i])
#         distance_matrix[i][m1] = distance_matrix[m1][i]
#
# # Set the distance_matrix for the merged cluster to a large value to indicate they are no longer separate clusters
# for i in range(len(data)):
#     distance_matrix[m2][i] = 1000
#     distance_matrix[i][m2] = 1000
#
# # Print the updated distance matrix
# print("Updated Distance Matrix:")
# for i in range(len(data)):
#     for j in range(len(data)):
#         if (j <= i) and distance_matrix[i][j]!= 1000:
#             print(distance_matrix[i][j], end='\t')
#     print('\t')
#
# # Dendrogram structure
# dendrogram = []
#
# # Perform single linkage hierarchical clustering
# while len(clusters) > 1:
#     min_distance = float('inf')
#     merge_clusters = None
#
#     # Find the two clusters with the minimum distance
#     for i in range(len(clusters)):
#         for j in range(i + 1, len(clusters)):
#             cluster1 = clusters[i]
#             cluster2 = clusters[j]
#             distance = min(distance_matrix[p1][p2] for p1 in cluster1 for p2 in cluster2)
#             if distance < min_distance:
#                 min_distance = distance
#                 merge_clusters = (i, j)
#
#     # Merge the clusters
#     i, j = merge_clusters
#     clusters[i].update(clusters[j])
#     del clusters[j]
#
#     # Append the cluster linkage to the dendrogram
#     dendrogram.append([i, j, min_distance, len(clusters[i]) + len(clusters[j])])
#
# # Plot the dendrogram
# plt.figure(figsize=(10, 5))
# plt.title('Single Linkage Dendrogram')
# plt.xlabel('Data Points')
# plt.ylabel('Distance')
# dendrogram = np.array(dendrogram)
# plt.plot(range(len(dendrogram)), dendrogram[:, 2])
# plt.show()
