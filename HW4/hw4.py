import csv
from json import load
import numpy as np
import sys
import matplotlib.pyplot as plt
import time

from numpy.linalg import norm
from scipy.cluster import hierarchy


def load_data(filepath):
    output = list()
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            output.append({
                "HP": row["HP"],
                "Attack": row["Attack"],
                "Defense": row["Defense"],
                "Sp. Atk": row["Sp. Atk"],
                "Sp. Def": row["Sp. Def"],
                "Speed": row["Speed"]
            })  # Remove all keys that aren't needed
    return output


def calc_features(row):
    return np.array([row["Attack"], row["Sp. Atk"], row["Speed"], row["Defense"], row["Sp. Def"], row["HP"]],
                    dtype='int64')

"""
Calculates distance of farthest 2 vectors in each cluster.
"""
def _calculate_distance(cluster1, cluster2):
    current_max_dist = -1
    for v1 in cluster1:
        for v2 in cluster2:
            current_max_dist = max(norm(v1 - v2), current_max_dist)
    return current_max_dist

def hac(features):
    clusters = dict()
    distance_matrix = dict()

    matrix = [[None for x in range(4)] for y in range(len(features) - 1)]  # This is our hac matrix n-1 x 4

    for i, vector in enumerate(features):
        clusters[i] = [vector]
    
    for id1, cluster1 in clusters.items():
        for id2, cluster2 in clusters.items():
            if id1 < id2:
                distance_matrix[(min(id1, id2), max(id1, id2))] = _calculate_distance(cluster1, cluster2)

    for rowIndex, row in enumerate(matrix):
        closest_pair = (sys.maxsize, sys.maxsize, sys.maxsize) # DIST, I, J

        for pair, distance in distance_matrix.items():
            closest_pair = min(closest_pair, (distance,) + pair)
        
        matrix[rowIndex][0] = closest_pair[1]
        matrix[rowIndex][1] = closest_pair[2]

        matrix[rowIndex][2] = closest_pair[0]

        # Delete the old keys and now merge the two clusters we've found.
        clusters[rowIndex + len(features)] = clusters[closest_pair[1]] + clusters[closest_pair[2]]
        del clusters[closest_pair[1]]
        del clusters[closest_pair[2]]

        # Store the length of our new cluster
        new_index = rowIndex + len(features)
        matrix[rowIndex][3] = len(clusters[new_index])

        # Update our distance matrix with this new cluster we've formed.
        for pair in list(distance_matrix):
            #We need to remove any pairs that use previously deleted clusters
            if closest_pair[1] in pair or closest_pair[2] in pair:
                del distance_matrix[pair]

        # Update distance_matrix to now include distances with our new cluster!
        for id, cluster in clusters.items():
            if id != new_index:
                distance_matrix[(min(id, new_index), max(id, new_index))] = _calculate_distance(cluster, clusters[new_index])

    return np.array(matrix)

def imshow_hac(Z):
    hierarchy.dendrogram(Z)
    plt.show()

imshow_hac(hac([calc_features(row) for row in load_data('Pokemon.csv')][:600]))


# DEPRECATED
# def hac_working(features):
#     clusters = dict()
#     matrix = [[None for x in range(4)] for y in range(len(features) - 1)]  # This is our hac matrix n-1 x 4

#     # Numbers each vector with a number from 0-(n-1) for their original cluster numbers
#     for i, vector in enumerate(features):
#         clusters[i] = [vector]

#     for rowIndex, row in enumerate(matrix):
#         # The way we calculate the distance is by subtracting 2 vectors and calculating the norm between the two.
#         # This will give us the distance at maximum between the two vectors farthest points. We then just store only the
#         # smallest distance we computed into our closest_pair variable. We then place those into our cluster dict as a
#         # new cluster pair using the formula n + rowIndex.
#         closest_pair = None
#         current_min_dist = sys.maxsize
        
#         for id1, cluster1 in clusters.items():
#             for id2, cluster2 in clusters.items():
#                 if id1 != id2:
#                     temp_dist = _calculate_distance(cluster1, cluster2)

#                     if temp_dist < current_min_dist:
#                         current_min_dist = temp_dist
#                         closest_pair = sorted(((id1, cluster1), (id2, cluster2)))
#                     elif temp_dist == current_min_dist:
#                         if closest_pair:
#                             temp_pair = sorted(((id1, cluster1), (id2, cluster2)))
                            
#                             if temp_pair[0][0] < closest_pair[0][0] or temp_pair[1][0] < closest_pair[1][0]:
#                                 closest_pair = temp_pair
#         # Merging Step

#         # Set the first two indices to their respective clusters.
#         matrix[rowIndex][0] = closest_pair[0][0]
#         matrix[rowIndex][1] = closest_pair[1][0]

#         # Place the linkage distance into the matrix
#         matrix[rowIndex][2] = current_min_dist

#         # Delete the old keys and now merge the two clusters we've found.
#         del clusters[closest_pair[0][0]]
#         del clusters[closest_pair[1][0]]
#         clusters[rowIndex + len(features)] = closest_pair[0][1] + closest_pair[1][1]

#         # Store the length of our new cluster
#         matrix[rowIndex][3] = len(clusters[rowIndex + len(features)])

#         print(f"Iteration: {rowIndex + 1}")
#     return np.array(matrix)