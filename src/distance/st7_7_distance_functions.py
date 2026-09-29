# distance functions
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
import time



# Distance between datasets by MMD by Jaccard index 
def mmd_from_jaccard(X, Y=None):
    # Jacard index
    jaccard_XX = 1 - pairwise_distances(X, X, metric='jaccard')
    jaccard_XY = 1 - pairwise_distances(X, Y, metric='jaccard')
    jaccard_YY = 1 - pairwise_distances(Y, Y, metric='jaccard')

    np.fill_diagonal(jaccard_XX, 0)
    np.fill_diagonal(jaccard_YY, 0)

    m_X = len(X)
    m_Y = len(Y)

    # Calculate terms
    term_X = np.sum(jaccard_XX) / (m_X * (m_X - 1))
    term_XY = np.sum(jaccard_XY) / (m_X * m_Y)
    term_Y = np.sum(jaccard_YY) / (m_Y * (m_Y - 1))

    mmd_squared = term_X - 2 * term_XY + term_Y
    mmd = np.sqrt(max(mmd_squared, 0)) # Ensure non-negative MMD

    return mmd


# Distance between dataset labels by MMD using RBF kernel
def mmd_from_rbf(X, Y=None, gamma=1.0):
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()



# Cosine distance between datasets after PCA dimensionality reduction
def cosine_distance_after_pca(X, Y=None, n_components=10):
    X_Y_combined = np.vstack([X, Y]) if Y is not None else X
    X_Y_scaled = StandardScaler().fit_transform(X_Y_combined)

    pca = PCA(n_components=n_components).fit(X_Y_scaled)
    X_pca = pca.transform(X_Y_scaled[:len(X)])
    Y_pca = pca.transform(X_Y_scaled[len(X):])

    return cosine_distances(X_pca, Y_pca).mean()


# Euclidean distance between datasets after PCA dimensionality reduction
def euclidean_distance_after_pca(X, Y=None, n_components=10):
    X_Y_combined = np.vstack([X, Y]) if Y is not None else X
    X_Y_scaled = StandardScaler().fit_transform(X_Y_combined)

    pca = PCA(n_components=n_components).fit(X_Y_scaled)
    X_pca = pca.transform(X_Y_scaled[:len(X)])
    Y_pca = pca.transform(X_Y_scaled[len(X):])

    return euclidean_distances(X_pca, Y_pca).mean()


# Cosine distance between datasets after UMAP dimensionality reduction
def cosine_distance_after_umap(X, Y=None, n_components=10, n_neighbors=15, min_dist=0.5):
    X_Y_combined = np.vstack([X, Y]) if Y is not None else X
    X_Y_scaled = StandardScaler().fit_transform(X_Y_combined)
    umap_model = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    X_Y_umap = umap_model.fit_transform(X_Y_scaled)

    X_umap = X_Y_umap[:len(X)]
    Y_umap = X_Y_umap[len(X):]

    return cosine_distances(X_umap, Y_umap).mean()



# Euclidean distance between datasets after UMAP dimensionality reduction
def euclidean_distance_after_umap(X, Y=None, n_components=10, n_neighbors=15, min_dist=0.5):
    X_Y_combined = np.vstack([X, Y]) if Y is not None else X
    X_Y_scaled = StandardScaler().fit_transform(X_Y_combined)
    umap_model = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    X_Y_umap = umap_model.fit_transform(X_Y_scaled)

    X_umap = X_Y_umap[:len(X)]
    Y_umap = X_Y_umap[len(X):]

    return euclidean_distances(X_umap, Y_umap).mean()



# Compute all the distances between two datasets
def compute_distance(X, Y):
    result = {}
    runtimes = {}

    # MMD Jaccard
    start = time.time()
    result["mmd_jaccard"] = mmd_from_jaccard(X, Y)
    runtimes["mmd_jaccard_runtime"] = time.time() - start

    '''
    # PCA Cosine
    start = time.time()
    result["pca_cosine"] = cosine_distance_after_pca(X, Y)
    runtimes["pca_cosine_runtime"] = time.time() - start

    # PCA Euclidean
    start = time.time()
    result["pca_euclidean"] = euclidean_distance_after_pca(X, Y)
    runtimes["pca_euclidean_runtime"] = time.time() - start

    # UMAP Cosine
    start = time.time()
    result["umap_cosine"] = cosine_distance_after_umap(X, Y)
    runtimes["umap_cosine_runtime"] = time.time() - start

    # UMAP Euclidean
    start = time.time()
    result["umap_euclidean"] = euclidean_distance_after_umap(X, Y)
    runtimes["umap_euclidean_runtime"] = time.time() - start
    '''

    return result, runtimes


def compute_label_distance(X, Y):
    start = time.time()
    result = {"mmd_rbf": mmd_from_rbf(X, Y)}
    runtime = {"mmd_rbf_runtime": time.time() - start}
    return result, runtime