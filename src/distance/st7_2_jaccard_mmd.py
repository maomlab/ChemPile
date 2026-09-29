import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

# Jaccard Kernel function
def jaccard_kernel(X, Y=None):
    # self kernel
    if Y is None:
        Y = X

    jaccard_distance = pairwise_distances(X, Y, metric='jaccard')
    jaccard_similarity = 1- jaccard_distance
    return jaccard_similarity

# MMD function
def compute_mmd_with_kernel(X, Y):
    # Step 1: Compute the Jaccard kernel
    similarity_matrix_X = jaccard_kernel(X)
    similarity_matrix_XY = jaccard_kernel(X, Y)
    similarity_matrix_Y = jaccard_kernel(Y)

    # To set the diganoal '0'
    # Added this on Jun 23th
    np.fill_diagonal(similarity_matrix_X, 0)
    np.fill_diagonal(similarity_matrix_Y, 0)

    # Step 2: Calculate MMD terms
    m_X = len(X)
    m_Y = len(Y)

    # Calculate terms
    term_X = np.sum(similarity_matrix_X) / (m_X * (m_X - 1))
    term_XY = np.sum(similarity_matrix_XY) / (m_X * m_Y)
    term_Y = np.sum(similarity_matrix_Y) / (m_Y * (m_Y - 1))

    # Step 3: Compute MMD^2
    mmd_squared = term_X - 2 * term_XY + term_Y

    # Step 4: Compute final MMD
    mmd = np.sqrt(mmd_squared)
    
    return mmd



