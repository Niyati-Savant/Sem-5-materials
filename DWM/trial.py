# import numpy as np
# import matplotlib.pyplot as plt
#
# data = np.array([[0.40, 0.53],
#                  [0.22, 0.38],
#                  [0.35, 0.32],
#                  [0.26, 0.19],
#                  [0.08, 0.41],
#                  [0.45, 0.30]])
#
# # Create clusters for each data point
# clusters = [{i} for i in range(len(data))]
# print(f"Initial clusters {clusters}")
#
# # Calculate Euclidean distance between two points
# def euclidean_distance(point1, point2):
#     X = np.sqrt(np.sum((point1 - point2) ** 2))
#     return round(X,2)
#
#
# # Create a distance matrix to store pairwise distances
# distance_matrix = np.zeros((len(data), len(data)))
#
# for i in range(len(data)):
#     for j in range(len(data)):
#         distance = euclidean_distance(data[i], data[j])
#         distance_matrix[i][j] = distance
#
# for i in range(len(data)):
#     for j in range(len(data)):
#         if (j<=i):
#             print(distance_matrix[i][j],end='\t')
#     print('\t')
#
# def merge_clusters(clusters, distance_matrix):
#     # Find the two clusters with the minimum distance
#
#     min_value = 100
#     m1, m2 = -1, -1
#
#     for i in range(len(clusters)):
#         for j in range(i + 1, len(clusters)):
#             if distance_matrix[i][j] < min_value:
#                 min_value = distance_matrix[i][j]
#                 m1, m2 = i, j
#     print(f"The min value is {min_value} between points {clusters[m1]} and {clusters[m2]}")
#     # Combine points m1 and m2 and update the clusters list
#     merged_cluster = clusters[m1].union(clusters[m2])
#     clusters[m1] = merged_cluster
#     clusters.pop(m2)
#     print(f"Now the clusters are {clusters}")
#
#     # Update the distance_matrix
#     for i in range(len(clusters)):
#         if i == m1:
#             continue
#         if i != m2:
#             distance_matrix[m1][i] = min(distance_matrix[m1][i], distance_matrix[m2][i])
#             distance_matrix[i][m1] = distance_matrix[m1][i]
#
#     # Set the distance_matrix for the merged cluster to a large value
#     for i in range(len(clusters)):
#         distance_matrix[m2][i] = 1000
#         distance_matrix[i][m2] = 1000
#
#     for i in range(len(clusters)):
#         for j in range(len(clusters)):
#             if (j <= i) and distance_matrix[i][j]!= 1000:
#                 print(distance_matrix[i][j], end='\t')
#         print('\t')
#     return clusters, distance_matrix
#
#
# # Continue merging clusters until only one cluster is left
# desired_num_clusters = 1
# for i in range(len(data)):
#     for j in range(len(data)):
#
#         distance = euclidean_distance(data[i], data[j])
#         distance_matrix[i][j] = distance
#
#
# while len(clusters) > desired_num_clusters:
#     clusters, distance_matrix = merge_clusters(clusters, distance_matrix)
#
# # Print the final clusters
# print("Final Clusters:")
# for i, cluster in enumerate(clusters):
#     print(f"Cluster {i}: {cluster}")
#
#
# from scipy.cluster.hierarchy import dendrogram, linkage
#
#
# linkage_matrix = linkage(data, method='merge_clusters')
#
# # Plot the dendrogram
# plt.figure(figsize=(8, 6))
# dendrogram(linkage_matrix)
# plt.title("Dendrogram")
# plt.xlabel("Data Point")
# plt.ylabel("Distance")
# plt.show()
#


from collections import Counter
# Define the transaction data
data = [
    ['T100', ['I1', 'I2', 'I5']],
    ['T200', ['I2', 'I4']],
    ['T300', ['I2', 'I3']],
    ['T400', ['I1', 'I2', 'I4']],
    ['T500', ['I1', 'I3']],
    ['T600', ['I2', 'I3']],
    ['T700', ['I1', 'I3']],
    ['T800', ['I1', 'I2', 'I3', 'I5']],
    ['T900', ['I1', 'I2', 'I3']]
]

# Set the minimum support threshold
min_support = 3

# Step 1: Find frequent 1-itemsets (C1 and L1)
C1 = Counter()
for transaction in data:
    for item in transaction[1]:
        C1[item] += 1

print("Frequent 1-Itemsets (L1):")
L1 = {frozenset([item]): support for item, support in C1.items() if support >= min_support}
for itemset, support in L1.items():
    print(f"{list(itemset)}: {support}")

# Step 2: Generate frequent itemsets of length 2 or more (Ck and Lk)
prev_Lk = L1
k = 2

while True:
    # Generate candidate itemsets Ck
    Ck = set()
    for itemset1 in prev_Lk:
        for itemset2 in prev_Lk:
            if len(itemset1.union(itemset2)) == k:
                Ck.add(itemset1.union(itemset2))

    # Count support for candidate itemsets
    Ck_support = Counter()
    for transaction in data:
        for itemset in Ck:
            if itemset.issubset(transaction[1]):
                Ck_support[itemset] += 1

    print(f"Frequent {k}-Itemsets (L{k}):")
    Lk = {itemset: support for itemset, support in Ck_support.items() if support >= min_support}
    for itemset, support in Lk.items():
        print(f"{list(itemset)}: {support}")

    if not Lk:
        break

    prev_Lk = Lk
    k += 1

# Step 3: Generate association rules
print("Association Rules:")
for frequent_itemset in Lk:
    if len(frequent_itemset) > 1:
        for item in frequent_itemset:
            antecedent = frozenset([item])
            consequent = frequent_itemset - antecedent

            antecedent_support = Ck_support[antecedent]
            consequent_support = Ck_support[consequent]
            confidence = consequent_support / antecedent_support * 100

            if confidence >= 100:
                print(f"{list(antecedent)} -> {list(consequent)} = {confidence}%")
