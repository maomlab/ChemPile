import numpy as np
from scipy.spatial.distance import pdist, squareform

def compute_mmd(X, Y):
    # Step 1: Measure Jaccard Distance
    # Calculate Jaccard distance for X_X (self distance for X)
    distance_matrix_X = pdist(X, 'jaccard')
    squareform_distance_matrix_X = squareform(distance_matrix_X)

    # Calculate Jaccard distance for X_Y (distance between X and Y)
    distance_matrix_XY = pdist(np.vstack([X, Y]), 'jaccard')
    squareform_distance_matrix_XY = squareform(distance_matrix_XY)

    # Calculate Jaccard distance for Y_Y (self distance for X)
    distance_matrix_Y = pdist(Y, 'jaccard')
    squareform_distance_matrix_Y = squareform(distance_matrix_Y)


    # Step 2: Convert Jaccard Distance to Similarity
    # Convert distance to similarity: similarity = 1 - distance
    similarity_matrix_X = 1 - squareform_distance_matrix_X 
    np.fill_diagonal(similarity_matrix_X, 1)  # Ensure the diagonal is 1 (self-similarity)

    similarity_matrix_XY = 1 - squareform_distance_matrix_XY 
    np.fill_diagonal(similarity_matrix_XY, 1)  # Ensure the diagonal is 1 (self-similarity)

    similarity_matrix_Y = 1 - squareform_distance_matrix_Y 
    np.fill_diagonal(similarity_matrix_Y, 1)  # Ensure the diagonal is 1 (self-similarity)


    # Step 3: Calculate MMD
    m_X = len(X)
    m_Y = len(Y)

    term_X = np.sum(similarity_matrix_X) / (m_X * (m_X - 1))
    term_XY = np.sum(similarity_matrix_XY) / (m_X * m_Y)
    term_Y = np.sum(similarity_matrix_Y) / (m_Y * (m_Y - 1))

    print("term_X is", term_X)
    print("term_XY is", term_XY)
    print("term_Y is", term_Y)

    # Compute MMD^2
    mmd_squared = term_X - 2 * term_XY + term_Y 

    print("mmd_squared is", mmd_squared)

    # Compute final MMD
    mmd = np.sqrt(mmd_squared)

    return mmd