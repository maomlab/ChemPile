import torch
import time

def pairwise_jaccard_similarity_gpu(A, B):
    """
    Compute Jaccard similarity between all pairs in A and B using GPU.
    A: (n_samples_a, n_features)
    B: (n_samples_b, n_features)
    Both A and B must be binary (0/1), dtype=torch.float32
    """

    # Compute the matrix product
    intersection = torch.matmul(A, B.T)  # (n_samples_a, n_samples_b)

    A_sum = A.sum(dim=1).unsqueeze(1)    # (n_samples_a, 1)
    B_sum = B.sum(dim=1).unsqueeze(0)    # (1, n_samples_b)
    union = A_sum + B_sum - intersection

    # Avoid division by zero
    similarity = intersection / (union + 1e-8)
    return similarity


def mmd_from_jaccard_gpu(X, Y):
    """
    MMD using Jaccard similarity (GPU version)
    X, Y: torch tensors on GPU, shape (n_samples, n_features)
    """
    X = X.float().to('cuda')
    Y = Y.float().to('cuda')

    sim_XX = pairwise_jaccard_similarity_gpu(X, X)
    sim_YY = pairwise_jaccard_similarity_gpu(Y, Y)
    sim_XY = pairwise_jaccard_similarity_gpu(X, Y)

    m = X.shape[0]
    n = Y.shape[0]

    # Remove diagonals (self-similarity)
    sim_XX.fill_diagonal_(0.0)
    sim_YY.fill_diagonal_(0.0)

    term_X = sim_XX.sum() / (m * (m - 1))
    term_Y = sim_YY.sum() / (n * (n - 1))
    term_XY = sim_XY.sum() / (m * n)

    mmd_squared = term_X - 2 * term_XY + term_Y
    mmd = torch.sqrt(torch.clamp(mmd_squared, min=0.0))

    return mmd.item()


def compute_distance_gpu(X, Y):
    result = {}
    runtimes = {}

    # MMD Jaccard
    start = time.time()
    result["mmd_jaccard"] = mmd_from_jaccard_gpu(X, Y)
    runtimes["mmd_jaccard_runtime"] = time.time() - start

    return result, runtimes
