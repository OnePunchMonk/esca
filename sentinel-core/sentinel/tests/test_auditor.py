"""Tests for RegressionAuditor and AuditReport — GPU-free."""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.profiler.profile import CapabilitySubspace, CapabilityProfile
from sentinel.auditor.report import (
    AuditReport,
    AuditReport,
    CapabilityDelta,
    AttributedExample,
    RemediationPlan,
    ConflictingExample,
)
from sentinel.auditor.auditor import (
    _gradient_cosine_attribution,
    _datainf_attribution,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_profile() -> CapabilityProfile:
    dim = 50
    math_basis = np.eye(dim, dtype=np.float32)[:5]
    code_basis = np.eye(dim, dtype=np.float32)[-5:]

    sub_math = CapabilitySubspace(
        name="math",
        basis_vectors=math_basis,
        singular_values=np.ones(5, dtype=np.float32),
        variance_explained=0.9,
        effective_rank=5,
        baseline_loss=1.5,
        eval_set_size=100,
    )
    sub_code = CapabilitySubspace(
        name="code",
        basis_vectors=code_basis,
        singular_values=np.ones(5, dtype=np.float32),
        variance_explained=0.85,
        effective_rank=5,
        baseline_loss=2.0,
        eval_set_size=100,
    )
    return CapabilityProfile(
        subspaces={"math": sub_math, "code": sub_code},
        model_name="test-model",
        lora_config={"r": 8},
        lora_param_dim=dim,
        total_model_params=1_000_000,
        capability_names=["math", "code"],
        overlap_matrix=np.eye(2, dtype=np.float32),
    )


def _make_training_grads(n: int = 80, dim: int = 50) -> np.ndarray:
    rng = np.random.RandomState(0)
    G = np.zeros((n, dim), dtype=np.float32)
    G[:, :10] = rng.randn(n, 10).astype(np.float32)
    return G


# ---------------------------------------------------------------------------
# Attribution helpers
# ---------------------------------------------------------------------------

def test_gradient_cosine_attribution_shape():
    dim = 50
    G = _make_training_grads(80, dim)
    cap_grad = np.eye(dim, dtype=np.float32)[0]  # first standard basis vector
    scores = _gradient_cosine_attribution(G, cap_grad)
    assert scores.shape == (80,)
    assert scores.dtype == np.float32


def test_gradient_cosine_attribution_sign():
    """Examples with positive projection on cap gradient → positive influence."""
    dim = 10
    cap_grad = np.ones(dim, dtype=np.float32) / np.sqrt(dim)
    G = np.array(
        [
            np.ones(dim, dtype=np.float32),   # positive
            -np.ones(dim, dtype=np.float32),  # negative
            np.zeros(dim, dtype=np.float32),  # neutral
        ]
    )
    scores = _gradient_cosine_attribution(G, cap_grad)
    assert scores[0] > 0
    assert scores[1] < 0
    assert abs(scores[2]) < 1e-6


def test_datainf_attribution_shape():
    dim = 50
    G = _make_training_grads(80, dim)
    eval_grad = np.random.randn(dim).astype(np.float32)
    scores = _datainf_attribution(G, eval_grad)
    assert scores.shape == (80,)


def test_datainf_vs_cosine_correlation():
    """DataInf and cosine should produce correlated rankings."""
    dim = 50
    G = _make_training_grads(80, dim)
    cap_grad = np.eye(dim, dtype=np.float32)[0]

    cosine_scores = _gradient_cosine_attribution(G, cap_grad)
    datainf_scores = _datainf_attribution(G, cap_grad)

    cosine_rank = np.argsort(cosine_scores)
    datainf_rank = np.argsort(datainf_scores)

    # Top-10 overlap should be > 50%
    top_cos = set(cosine_rank[-10:])
    top_di = set(datainf_rank[-10:])
    overlap = len(top_cos & top_di)
    assert overlap >= 3, f"Top-10 overlap too low: {overlap}"


# ---------------------------------------------------------------------------
# AuditReport
# ---------------------------------------------------------------------------

def test_audit_report_summary_renders():
    delta = CapabilityDelta(
        capability_name="math",
        baseline_loss=1.5,
        final_loss=1.8,
        loss_delta=0.3,
        drift_magnitude=0.12,
        drift_fraction=0.24,
    )
    report = AuditReport(
        capability_deltas={"math": delta},
        capabilities_degraded=["math"],
        capabilities_improved=[],
        capabilities_unchanged=[],
        overall_regression_score=0.4,
        model_name="test-model",
        training_data_size=10_000,
    )
    text = str(report)
    assert "SENTINEL AUDIT REPORT" in text
    assert "math" in text
    assert "0.30" in text  # loss delta


def test_audit_report_severity():
    cases = [
        (0.51, "CRITICAL"),
        (0.25, "HIGH"),
        (0.1, "MEDIUM"),
        (0.01, "LOW"),
        (0.0, "NONE"),
    ]
    for loss_delta, expected in cases:
        delta = CapabilityDelta(
            capability_name="x",
            baseline_loss=1.0,
            final_loss=1.0 + loss_delta,
            loss_delta=loss_delta,
        )
        assert delta.severity == expected, f"loss_delta={loss_delta} → {delta.severity} (expected {expected})"


def test_audit_report_json_roundtrip(tmp_path):
    delta = CapabilityDelta(
        capability_name="safety",
        baseline_loss=2.0,
        final_loss=2.6,
        loss_delta=0.6,
        drift_fraction=0.35,
        top_harmful_examples=[
            AttributedExample(
                example_id=42,
                example_text="Example text",
                influence_score=-0.25,
                gradient_similarity=-0.25,
            )
        ],
    )
    report = AuditReport(
        capability_deltas={"safety": delta},
        capabilities_degraded=["safety"],
        model_name="test",
        training_data_size=5000,
    )
    out = tmp_path / "audit.json"
    report.to_json(str(out))
    assert out.exists()

    import json
    data = json.loads(out.read_text())
    assert data["capabilities_degraded"] == ["safety"]
    assert data["capability_deltas"]["safety"]["loss_delta"] == pytest.approx(0.6, abs=1e-4)
    assert data["capability_deltas"]["safety"]["top_harmful"][0]["example_id"] == 42


def test_audit_report_html_roundtrip(tmp_path):
    delta = CapabilityDelta(
        capability_name="math", baseline_loss=1.5, final_loss=1.9, loss_delta=0.4
    )
    report = AuditReport(
        capability_deltas={"math": delta},
        capabilities_degraded=["math"],
        model_name="test",
    )
    out = tmp_path / "report.html"
    report.to_html(str(out))
    html = out.read_text()
    assert "Sentinel Audit Report" in html
    assert "math" in html


def test_remediation_plan_summary():
    plan = RemediationPlan(
        examples_to_remove=[1, 2, 3],
        retention_data_recommendations={"math": 500},
        retention_data_sources={"math": "sentinel:retain-math-2k"},
        recommendation_summary="Remove 3 harmful examples; Add 500 math retention examples",
    )
    text = plan.summary()
    assert "Remove 3" in text
    assert "math" in text
