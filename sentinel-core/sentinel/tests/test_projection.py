"""Tests for gradient projection."""

import numpy as np
import torch
import pytest

from sentinel.optimizer.projection import gradient_projection_step
from sentinel.utils.svd_utils import joint_projection_matrix


def test_projection_removes_protected_component():
    """Gradient along a protected subspace should be fully removed at β=1."""
    # Subspace = span of e1, e2 in R^5
    basis = np.eye(5, dtype=np.float32)[:2]  # (2, 5)
    gradient = torch.tensor([3.0, 4.0, 1.0, 2.0, 0.5])

    result = gradient_projection_step(
        gradient,
        protected_subspaces={"math": basis},
        betas={"math": 1.0},
    )
    # e1 and e2 components should be zero
    assert abs(result[0].item()) < 1e-5
    assert abs(result[1].item()) < 1e-5
    # Other components unchanged
    assert abs(result[2].item() - 1.0) < 1e-5
    assert abs(result[3].item() - 2.0) < 1e-5


def test_projection_partial_beta():
    """β=0.5 should remove half the protected component."""
    basis = np.eye(4, dtype=np.float32)[:1]  # span of e1
    gradient = torch.tensor([4.0, 0.0, 0.0, 0.0])

    result = gradient_projection_step(
        gradient,
        protected_subspaces={"safety": basis},
        betas={"safety": 0.5},
    )
    assert abs(result[0].item() - 2.0) < 1e-5


def test_projection_zero_beta_is_identity():
    """β=0 should leave gradient unchanged."""
    basis = np.eye(4, dtype=np.float32)[:2]
    gradient = torch.tensor([1.0, 2.0, 3.0, 4.0])

    result = gradient_projection_step(
        gradient,
        protected_subspaces={"code": basis},
        betas={"code": 0.0},
    )
    torch.testing.assert_close(result, gradient)


def test_joint_projection_avoids_double_removal():
    """Joint projection on overlapping subspaces shouldn't over-project."""
    # Both bases include e1
    basis_a = np.eye(4, dtype=np.float32)[:1]  # e1
    basis_b = np.eye(4, dtype=np.float32)[:1]  # e1 again
    joint = joint_projection_matrix([basis_a, basis_b])

    gradient = torch.tensor([4.0, 3.0, 0.0, 0.0])
    result = gradient_projection_step(
        gradient,
        protected_subspaces={"a": basis_a, "b": basis_b},
        betas={"a": 1.0, "b": 1.0},
        joint_basis=joint,
    )
    # e1 should be removed once, not twice
    assert abs(result[0].item()) < 1e-5
    assert abs(result[1].item() - 3.0) < 1e-5
