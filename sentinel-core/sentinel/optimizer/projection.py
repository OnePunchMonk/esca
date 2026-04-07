"""Gradient projection methods for capability protection."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torch import Tensor


def gradient_projection_step(
    gradient: Tensor,
    protected_subspaces: Dict[str, np.ndarray],
    betas: Dict[str, float],
    joint_basis: np.ndarray | None = None,
) -> Tensor:
    """Remove protected capability components from a gradient vector.

    Parameters
    ----------
    gradient : (d,) LoRA gradient as a flat tensor
    protected_subspaces : mapping of capability → (k, d) basis arrays
    betas : mapping of capability → protection strength in [0, 1]
    joint_basis : optional pre-computed joint basis (d, k_total) from
        ``joint_projection_matrix``.  When provided, a single joint
        projection is used instead of sequential per-capability projections,
        which avoids over-projection when subspaces overlap.

    Returns
    -------
    Protected gradient (d,) tensor.
    """
    g = gradient.clone().float()

    if joint_basis is not None:
        # Uniform β — use joint projection
        mean_beta = np.mean(list(betas.values()))
        Q = torch.from_numpy(joint_basis).float()
        proj = Q @ (Q.T @ g)
        g = g - mean_beta * proj
    else:
        # Per-capability sequential projection
        for cap_name, basis in protected_subspaces.items():
            beta = betas.get(cap_name, 0.0)
            if beta <= 0:
                continue
            V = torch.from_numpy(basis).float()  # (k, d)
            proj = V.T @ (V @ g)  # (d,)
            g = g - beta * proj

    return g
