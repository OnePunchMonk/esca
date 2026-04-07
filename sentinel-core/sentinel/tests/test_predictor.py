"""Tests for RegressionPredictor with pre-computed gradient matrices."""

import numpy as np
import pytest

from sentinel.profiler.profile import CapabilitySubspace, CapabilityProfile
from sentinel.predictor import RegressionPredictor, TrainingConfig


def _make_profile_and_training_grads():
    """Create a profile where 'math' subspace overlaps with training gradients
    but 'code' does not."""
    rng = np.random.RandomState(42)
    dim = 50

    # Math subspace: first 5 directions
    math_basis = np.eye(dim, dtype=np.float32)[:5]
    # Code subspace: last 5 directions
    code_basis = np.eye(dim, dtype=np.float32)[-5:]

    sub_math = CapabilitySubspace(
        name="math",
        basis_vectors=math_basis,
        singular_values=np.ones(5, dtype=np.float32),
        variance_explained=0.9,
        effective_rank=5,
        baseline_loss=1.0,
        eval_set_size=100,
    )
    sub_code = CapabilitySubspace(
        name="code",
        basis_vectors=code_basis,
        singular_values=np.ones(5, dtype=np.float32),
        variance_explained=0.9,
        effective_rank=5,
        baseline_loss=1.0,
        eval_set_size=100,
    )

    profile = CapabilityProfile(
        subspaces={"math": sub_math, "code": sub_code},
        model_name="test",
        lora_config={"r": 8},
        lora_param_dim=dim,
        total_model_params=1_000_000,
        capability_names=["math", "code"],
        overlap_matrix=np.eye(2, dtype=np.float32),
    )

    # Training gradients: entirely in the first 10 dims (overlapping math, not code)
    n_samples = 100
    G_train = np.zeros((n_samples, dim), dtype=np.float32)
    G_train[:, :10] = rng.randn(n_samples, 10).astype(np.float32)

    return profile, G_train


def test_predictor_identifies_high_risk_capability():
    profile, G_train = _make_profile_and_training_grads()
    predictor = RegressionPredictor(profile, training_config=TrainingConfig())
    risk = predictor.predict(G_train)

    assert risk.capabilities["math"].risk_level in ("HIGH", "CRITICAL", "MEDIUM")
    assert risk.capabilities["code"].risk_level in ("NONE", "LOW")
    assert risk.capabilities["math"].risk_score > risk.capabilities["code"].risk_score


def test_risk_report_str_renders():
    profile, G_train = _make_profile_and_training_grads()
    predictor = RegressionPredictor(profile)
    risk = predictor.predict(G_train)
    text = str(risk)
    assert "SENTINEL RISK REPORT" in text
    assert "math" in text
