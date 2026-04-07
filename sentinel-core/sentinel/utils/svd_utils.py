"""SVD and subspace geometry utilities."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def randomized_svd(
    matrix: Tensor | np.ndarray,
    rank: int,
    n_oversamples: int = 10,
    n_power_iter: int = 2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomized SVD (Halko et al. 2011) — compute a rank-k approximation
    without materialising the full decomposition.

    Parameters
    ----------
    matrix : (m, n) tensor or array  — the data matrix
    rank   : desired rank k
    n_oversamples : oversampling for accuracy (total sketch dim = rank + n_oversamples)
    n_power_iter  : power iterations for better approximation of top singular space

    Returns
    -------
    U  : (m, rank)   left singular vectors
    S  : (rank,)     singular values
    Vt : (rank, n)   right singular vectors
    """
    if isinstance(matrix, Tensor):
        matrix = matrix.float().numpy()

    rng = np.random.RandomState(seed)
    m, n = matrix.shape
    k = min(rank + n_oversamples, min(m, n))

    # Random projection
    omega = rng.randn(n, k).astype(np.float32)
    Y = matrix @ omega  # (m, k)

    # Power iteration for better approximation
    for _ in range(n_power_iter):
        Y = matrix @ (matrix.T @ Y)

    Q, _ = np.linalg.qr(Y)  # (m, k) orthonormal basis for column space

    # Project and compute exact SVD on the small matrix
    B = Q.T @ matrix  # (k, n)
    U_hat, S, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ U_hat  # (m, k)

    # Truncate to desired rank
    return U[:, :rank], S[:rank], Vt[:rank, :]


def subspace_overlap(basis_a: np.ndarray, basis_b: np.ndarray) -> float:
    """Compute the overlap (mean squared cosine similarity) between two subspaces.

    Each basis is (k, d) where rows are orthonormal basis vectors.
    Returns a scalar in [0, 1] where 1 = identical subspaces.
    """
    # Gram matrix of cross-products: (k_a, k_b)
    G = basis_a @ basis_b.T
    # Frobenius norm squared, normalised by the smaller subspace dimension
    k_min = min(basis_a.shape[0], basis_b.shape[0])
    if k_min == 0:
        return 0.0
    return float(np.sum(G ** 2) / k_min)


def project_onto_subspace(vector: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project *vector* (d,) onto the subspace spanned by *basis* (k, d).

    Returns the projection component (d,).
    """
    # coefficients = basis @ vector  → (k,)
    coeffs = basis @ vector
    return basis.T @ coeffs  # (d,)


def joint_projection_matrix(bases: list[np.ndarray]) -> np.ndarray:
    """Compute the projector onto the union of multiple subspaces.

    Uses QR on the concatenated bases so that overlapping subspaces are
    handled correctly (no double-projection).

    Parameters
    ----------
    bases : list of (k_i, d) arrays

    Returns
    -------
    P : (d, d) orthogonal projection matrix  (or equivalently the Q factor
        that satisfies P = Q Q^T).  For memory efficiency we return Q of
        shape (d, k_total) so the caller can do ``Q @ (Q.T @ g)`` instead
        of materialising the full (d, d) matrix.
    """
    if not bases:
        raise ValueError("Need at least one basis.")
    concatenated = np.vstack(bases)  # (sum_k_i, d)
    Q, R = np.linalg.qr(concatenated.T)  # (d, sum_k_i) — columns are orthonormal
    # Only keep columns with significant R diagonal (effective rank)
    diag = np.abs(np.diag(R))
    tol = max(diag) * 1e-8 if len(diag) > 0 else 1e-8
    effective_rank = int((diag > tol).sum())
    return Q[:, :effective_rank]
