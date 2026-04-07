"""Tests for SVD and subspace utilities."""

import numpy as np
import pytest

from sentinel.utils.svd_utils import (
    randomized_svd,
    subspace_overlap,
    project_onto_subspace,
    joint_projection_matrix,
)


def test_randomized_svd_recovers_low_rank_matrix():
    """A rank-3 matrix should be well-approximated by rank-3 rSVD."""
    rng = np.random.RandomState(0)
    A = rng.randn(50, 3).astype(np.float32)
    B = rng.randn(3, 100).astype(np.float32)
    M = A @ B  # rank 3

    U, S, Vt = randomized_svd(M, rank=3)
    approx = U @ np.diag(S) @ Vt
    rel_error = np.linalg.norm(M - approx) / np.linalg.norm(M)
    assert rel_error < 0.01


def test_subspace_overlap_identical_subspaces():
    """Overlap of a subspace with itself should be 1."""
    rng = np.random.RandomState(1)
    Q, _ = np.linalg.qr(rng.randn(10, 3).T)  # (3, 10) orthonormal rows
    # Actually qr returns (10,3) so let's do it right
    basis = np.linalg.qr(rng.randn(10, 3))[0].T[:3]  # (3, 10)
    overlap = subspace_overlap(basis, basis)
    assert abs(overlap - 1.0) < 1e-5


def test_subspace_overlap_orthogonal_subspaces():
    """Overlap of orthogonal subspaces should be ~0."""
    # Use first 3 and last 3 standard basis vectors in R^10
    basis_a = np.eye(10, dtype=np.float32)[:3]  # (3, 10)
    basis_b = np.eye(10, dtype=np.float32)[7:]  # (3, 10)
    overlap = subspace_overlap(basis_a, basis_b)
    assert overlap < 1e-5


def test_project_onto_subspace():
    """Projection onto e1-e2 subspace should zero out other components."""
    basis = np.eye(5, dtype=np.float32)[:2]  # span of e1, e2
    v = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    proj = project_onto_subspace(v, basis)
    np.testing.assert_allclose(proj, [1, 2, 0, 0, 0], atol=1e-6)


def test_joint_projection_matrix_covers_union():
    """Joint projector should cover both subspaces."""
    basis_a = np.eye(6, dtype=np.float32)[:2]  # e1, e2
    basis_b = np.eye(6, dtype=np.float32)[2:4]  # e3, e4
    Q = joint_projection_matrix([basis_a, basis_b])  # (6, 4)

    v = np.array([1, 1, 1, 1, 1, 1], dtype=np.float32)
    proj = Q @ (Q.T @ v)
    # Should project onto e1-e4 span, zeroing e5 and e6
    np.testing.assert_allclose(proj[:4], [1, 1, 1, 1], atol=1e-5)
    np.testing.assert_allclose(proj[4:], [0, 0], atol=1e-5)
