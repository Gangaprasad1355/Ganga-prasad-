# Ganga-prasad-import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import cosine_similarity

# Euclidean Distance
def euclidean_distance(vec1, vec2):
    return euclidean(vec1, vec2)

# Cosine Similarity
def cosine_similarity_measure(vec1, vec2):
    vec1 = np.array(vec1).reshape(1, -1)
    vec2 = np.array(vec2).reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]

# Jaccard Similarity
def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

# Test data
vec1 = [1, 2, 3]
vec2 = [4, 5, 6]

set1 = set([1, 2, 3, 4])
set2 = set([3, 4, 5, 6])

# Compute similarity measures
print("Euclidean Distance:", euclidean_distance(vec1, vec2))
print("Cosine Similarity:", cosine_similarity_measure(vec1, vec2))
print("Jaccard Similarity:", jaccard_similarity(set1, set2))
