"""Tests for CapabilityProfile serialization."""

import tempfile

import numpy as np
import pytest

from sentinel.profiler.profile import CapabilitySubspace, CapabilityProfile


def _make_profile() -> CapabilityProfile:
    sub_math = CapabilitySubspace(
        name="math",
        basis_vectors=np.random.randn(8, 100).astype(np.float32),
        singular_values=np.sort(np.random.rand(8).astype(np.float32))[::-1],
        variance_explained=0.85,
        effective_rank=6,
        baseline_loss=1.23,
        eval_set_size=50,
    )
    sub_code = CapabilitySubspace(
        name="code",
        basis_vectors=np.random.randn(8, 100).astype(np.float32),
        singular_values=np.sort(np.random.rand(8).astype(np.float32))[::-1],
        variance_explained=0.79,
        effective_rank=5,
        baseline_loss=1.45,
        eval_set_size=40,
    )
    overlap = np.array([[1.0, 0.3], [0.3, 1.0]], dtype=np.float32)
    return CapabilityProfile(
        subspaces={"math": sub_math, "code": sub_code},
        model_name="test-model",
        lora_config={"r": 16, "lora_alpha": 32},
        lora_param_dim=100,
        total_model_params=7_000_000_000,
        overlap_matrix=overlap,
        capability_names=["math", "code"],
    )


def test_save_load_roundtrip():
    profile = _make_profile()
    with tempfile.TemporaryDirectory() as tmpdir:
        profile.save(tmpdir)
        loaded = CapabilityProfile.load(tmpdir)

    assert loaded.model_name == "test-model"
    assert set(loaded.capability_names) == {"math", "code"}
    assert loaded.lora_param_dim == 100
    np.testing.assert_allclose(
        loaded.subspaces["math"].basis_vectors,
        profile.subspaces["math"].basis_vectors,
        atol=1e-5,
    )
    np.testing.assert_allclose(loaded.overlap_matrix, profile.overlap_matrix, atol=1e-5)
    assert loaded.subspaces["math"].effective_rank == 6


def test_summary_contains_capability_names():
    profile = _make_profile()
    s = profile.summary()
    assert "math" in s
    assert "code" in s
    assert "test-model" in s


def test_overlap_report():
    profile = _make_profile()
    report = profile.overlap_report()
    assert "math" in report
    assert "code" in report
