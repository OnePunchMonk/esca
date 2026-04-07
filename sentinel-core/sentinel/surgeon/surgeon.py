"""DataSurgeon — create SurgeryPlans from AuditReport results."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional

import numpy as np

from ..auditor.report import AuditReport, CapabilityDelta
from ..profiler.profile import CapabilityProfile
from .plan import SurgeryPlan

logger = logging.getLogger("sentinel.surgeon")

# Built-in retention dataset registry (symbolic handles shipped with sentinel)
_RETENTION_SOURCES = {
    "math": "sentinel:retain-math-2k",
    "code": "sentinel:retain-code-2k",
    "safety": "sentinel:retain-safety-1k",
    "reasoning": "sentinel:retain-reasoning-2k",
    "factual": "sentinel:retain-factual-2k",
    "instruction": "sentinel:retain-instruct-2k",
}


class DataSurgeon:
    """Convert an :class:`~sentinel.auditor.AuditReport` into a :class:`SurgeryPlan`.

    Usage::

        surgeon = DataSurgeon(audit_report=report, profile=profile)
        plan = surgeon.plan(training_data=sft_dataset)
        cleaned = plan.apply(sft_dataset)
    """

    def __init__(
        self,
        audit_report: AuditReport,
        profile: CapabilityProfile,
        *,
        strategy: Literal["remove", "reweight", "augment", "smart"] = "smart",
        max_removals: int = 500,
        max_target_task_cost: float = 0.02,
        reweight_lambda: float = 2.0,
        augment_with_retention: bool = True,
        retention_budget: int = 1000,
        retention_source: str = "sentinel:standard",
    ) -> None:
        self.report = audit_report
        self.profile = profile
        self.strategy = strategy
        self.max_removals = max_removals
        self.max_target_task_cost = max_target_task_cost
        self.reweight_lambda = reweight_lambda
        self.augment = augment_with_retention
        self.retention_budget = retention_budget
        self.retention_source = retention_source

    def plan(self, training_data: Any = None) -> SurgeryPlan:
        """Compute a :class:`SurgeryPlan` from the audit report.

        Parameters
        ----------
        training_data
            The original training dataset (optional — used for size info).
        """
        n_train = len(training_data) if training_data is not None and hasattr(training_data, "__len__") else 0

        degraded = {
            name: delta
            for name, delta in self.report.capability_deltas.items()
            if delta.regressed and delta.severity not in ("NONE",)
        }

        if not degraded:
            logger.info("No significant regressions detected — empty surgery plan.")
            return SurgeryPlan(summary_text="No action required — model is healthy.")

        # ------------------------------------------------------------------
        # Compute removal candidates from harmful examples
        # ------------------------------------------------------------------
        harmful_scores: Dict[int, float] = {}  # example_id → mean negative influence
        for cap_name, delta in degraded.items():
            for ex in delta.top_harmful_examples:
                if ex.influence_score < 0:
                    if ex.example_id not in harmful_scores:
                        harmful_scores[ex.example_id] = 0.0
                    harmful_scores[ex.example_id] += ex.influence_score

        # Sort by most harmful (most negative)
        sorted_harmful = sorted(harmful_scores.items(), key=lambda kv: kv[1])
        examples_to_remove = [eid for eid, _ in sorted_harmful[: self.max_removals]]

        # ------------------------------------------------------------------
        # Per-example reweighting (strategy reweight / smart)
        # ------------------------------------------------------------------
        example_weights: Dict[int, float] = {}

        if self.strategy in ("reweight", "smart"):
            for cap_name, delta in degraded.items():
                for ex in delta.top_harmful_examples:
                    eid = ex.example_id
                    harm = abs(min(ex.influence_score, 0))
                    # w = 1 / (1 + λ · harm)  — higher harm → lower weight
                    new_w = 1.0 / (1.0 + self.reweight_lambda * harm)
                    if eid in example_weights:
                        example_weights[eid] = min(example_weights[eid], new_w)
                    else:
                        example_weights[eid] = new_w

        # ------------------------------------------------------------------
        # Retention augmentation
        # ------------------------------------------------------------------
        retention_requests: Dict[str, int] = {}

        if self.augment:
            for cap_name, delta in degraded.items():
                if delta.severity in ("HIGH", "CRITICAL"):
                    budget = min(self.retention_budget, 1000)
                elif delta.severity == "MEDIUM":
                    budget = min(self.retention_budget // 2, 500)
                else:
                    budget = min(self.retention_budget // 4, 200)
                retention_requests[cap_name] = budget

        # ------------------------------------------------------------------
        # Predicted outcomes
        # ------------------------------------------------------------------
        recovery: Dict[str, float] = {}
        for cap_name, delta in degraded.items():
            # Heuristic: removing top harmful examples recovers ~60% of loss
            n_removed_for_cap = sum(
                1 for ex in delta.top_harmful_examples if ex.example_id in examples_to_remove
            )
            recovery_frac = min(0.9, n_removed_for_cap * 0.05 + (retention_requests.get(cap_name, 0) * 0.0003))
            recovery[cap_name] = recovery_frac

        target_cost = (
            len(examples_to_remove) / max(n_train, 1)
            if self.strategy in ("remove", "smart")
            else 0.0
        )

        # ------------------------------------------------------------------
        # Smart strategy: prefer reweighting over hard removal when overlap
        # with target task is high (proxied by n_conflicting_examples)
        # ------------------------------------------------------------------
        if self.strategy == "smart":
            n_conflicting = len(self.report.conflicting_examples)
            if n_conflicting > 100:
                # Many conflicting examples — hard removal is costly, prefer reweight
                examples_to_remove = []
                logger.info(
                    "Smart strategy: %d conflicting examples → preferring reweighting over removal.",
                    n_conflicting,
                )

        # ------------------------------------------------------------------
        # Summary text
        # ------------------------------------------------------------------
        actions = []
        if examples_to_remove:
            actions.append(f"remove {len(examples_to_remove)} harmful examples")
        if example_weights:
            actions.append(f"reweight {len(example_weights)} examples")
        if retention_requests:
            caps = ", ".join(
                f"{cap} (+{n})" for cap, n in retention_requests.items()
            )
            actions.append(f"augment with retention data: {caps}")
        summary_text = "; ".join(actions) if actions else "No action required"

        return SurgeryPlan(
            examples_to_remove=examples_to_remove,
            example_weights=example_weights,
            retention_requests=retention_requests,
            summary_text=summary_text,
            predicted_regression_recovery=recovery,
            predicted_target_task_cost=target_cost,
        )
