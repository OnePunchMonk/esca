"""Tests for DataSurgeon and SurgeryPlan."""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.profiler.profile import CapabilitySubspace, CapabilityProfile
from sentinel.auditor.report import (
    AuditReport,
    CapabilityDelta,
    AttributedExample,
    ConflictingExample,
    RemediationPlan,
)
from sentinel.surgeon import DataSurgeon, SurgeryPlan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_profile() -> CapabilityProfile:
    dim = 30
    sub = CapabilitySubspace(
        name="math",
        basis_vectors=np.eye(dim, dtype=np.float32)[:5],
        singular_values=np.ones(5, dtype=np.float32),
        variance_explained=0.9,
        effective_rank=5,
        baseline_loss=1.5,
        eval_set_size=100,
    )
    return CapabilityProfile(
        subspaces={"math": sub},
        model_name="test",
        lora_config={"r": 8},
        lora_param_dim=dim,
        total_model_params=500_000,
        capability_names=["math"],
        overlap_matrix=np.eye(1, dtype=np.float32),
    )


def _make_audit_report(has_regression: bool = True) -> AuditReport:
    harmful = [
        AttributedExample(
            example_id=i,
            example_text=f"Example {i}",
            influence_score=-(i + 1) * 0.1,
            gradient_similarity=-(i + 1) * 0.1,
        )
        for i in range(5)
    ]
    if has_regression:
        delta = CapabilityDelta(
            capability_name="math",
            baseline_loss=1.5,
            final_loss=2.1,
            loss_delta=0.6,
            top_harmful_examples=harmful,
        )
        return AuditReport(
            capability_deltas={"math": delta},
            capabilities_degraded=["math"],
            overall_regression_score=0.7,
            model_name="test",
            training_data_size=1000,
        )
    else:
        delta = CapabilityDelta(
            capability_name="math",
            baseline_loss=1.5,
            final_loss=1.48,
            loss_delta=-0.02,
        )
        return AuditReport(
            capability_deltas={"math": delta},
            capabilities_improved=["math"],
            overall_regression_score=0.0,
            model_name="test",
            training_data_size=1000,
        )


# ---------------------------------------------------------------------------
# SurgeryPlan
# ---------------------------------------------------------------------------

def test_surgery_plan_summary_renders():
    plan = SurgeryPlan(
        examples_to_remove=[1, 2, 3],
        retention_requests={"math": 500},
        summary_text="Remove 3 examples; augment math",
        predicted_target_task_cost=0.003,
        predicted_regression_recovery={"math": 0.7},
    )
    text = str(plan)
    assert "SURGERY PLAN" in text
    assert "math" in text


def test_surgery_plan_remove_harmful_list():
    plan = SurgeryPlan(examples_to_remove=[0, 2, 4])
    data = ["a", "b", "c", "d", "e"]
    result = plan.remove_harmful(data)
    assert result == ["b", "d"]


def test_surgery_plan_json_roundtrip(tmp_path):
    plan = SurgeryPlan(
        examples_to_remove=[10, 20],
        example_weights={5: 0.3, 6: 0.7},
        retention_requests={"math": 200},
        summary_text="test plan",
        predicted_target_task_cost=0.01,
    )
    path = str(tmp_path / "plan.json")
    plan.to_json(path)
    loaded = SurgeryPlan.from_json(path)
    assert loaded.examples_to_remove == [10, 20]
    assert loaded.example_weights[5] == pytest.approx(0.3)
    assert loaded.retention_requests["math"] == 200


# ---------------------------------------------------------------------------
# DataSurgeon
# ---------------------------------------------------------------------------

def test_surgeon_no_regression_returns_empty_plan():
    profile = _make_profile()
    report = _make_audit_report(has_regression=False)
    surgeon = DataSurgeon(report, profile)
    plan = surgeon.plan()
    # No regression → no harmful examples to remove
    assert plan.examples_to_remove == []
    assert "No action required" in plan.summary_text


def test_surgeon_regression_removes_harmful_examples():
    profile = _make_profile()
    report = _make_audit_report(has_regression=True)
    surgeon = DataSurgeon(report, profile, strategy="remove", max_removals=10)
    plan = surgeon.plan()
    # Should flag the harmful examples (IDs 0-4, all with neg influence)
    assert len(plan.examples_to_remove) > 0


def test_surgeon_augments_degraded_capabilities():
    profile = _make_profile()
    report = _make_audit_report(has_regression=True)
    surgeon = DataSurgeon(report, profile, augment_with_retention=True)
    plan = surgeon.plan()
    # Should request retention data for 'math'
    assert "math" in plan.retention_requests
    assert plan.retention_requests["math"] > 0


def test_surgeon_reweight_strategy():
    profile = _make_profile()
    report = _make_audit_report(has_regression=True)
    surgeon = DataSurgeon(report, profile, strategy="reweight")
    plan = surgeon.plan()
    # Reweight strategy should populate example_weights
    assert len(plan.example_weights) > 0
    for w in plan.example_weights.values():
        assert 0 < w <= 1.0


def test_surgeon_smart_strategy_with_many_conflicting():
    """When many conflicting examples exist, smart strategy prefers reweighting."""
    profile = _make_profile()
    report = _make_audit_report(has_regression=True)
    # Inject many conflicting examples
    report.conflicting_examples = [
        ConflictingExample(
            example_id=i,
            example_text=f"c{i}",
            positive_effects={"customer_support": 0.5},
            negative_effects={"math": -0.3},
            net_value=-0.2,
            recommendation="REWEIGHT",
        )
        for i in range(200)
    ]
    surgeon = DataSurgeon(report, profile, strategy="smart")
    plan = surgeon.plan()
    # Smart strategy should prefer reweighting — examples_to_remove should be empty
    assert plan.examples_to_remove == []
