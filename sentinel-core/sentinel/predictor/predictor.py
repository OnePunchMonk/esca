"""RegressionPredictor — predict capability regression before training."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import Tensor

from ..profiler.profile import CapabilityProfile
from ..utils.lora_utils import extract_lora_gradient
from ..utils.svd_utils import randomized_svd, subspace_overlap
from .risk_report import (
    CapabilityRisk,
    ExampleRisk,
    RiskReport,
    TrainingConfig,
)

logger = logging.getLogger("sentinel.predictor")

# Thresholds for risk levels (subspace overlap)
_RISK_THRESHOLDS = {
    "CRITICAL": 0.40,
    "HIGH": 0.25,
    "MEDIUM": 0.10,
    "LOW": 0.03,
}


def _risk_level(overlap: float) -> str:
    for level, threshold in _RISK_THRESHOLDS.items():
        if overlap >= threshold:
            return level
    return "NONE"


def _overlap_to_predicted_delta(
    overlap: float,
    learning_rate: float,
    num_epochs: int,
    baseline_loss: float,
) -> float:
    """Simple calibration: map subspace overlap → predicted accuracy Δ%.

    This is a first-order approximation.  A proper calibration model
    (fit on hundreds of runs) replaces this in later iterations.
    """
    # Higher overlap, higher LR, more epochs → more regression
    intensity = overlap * (learning_rate / 2e-5) * (num_epochs / 3)
    # Cap at reasonable bounds
    delta_pct = -intensity * 30.0  # rough scale factor
    return max(delta_pct, -50.0)


class RegressionPredictor:
    """Predict which capabilities will regress given training data.

    Usage::

        predictor = RegressionPredictor(profile)
        risk = predictor.predict(training_data)
        print(risk)  # pretty risk report
    """

    def __init__(
        self,
        profile: CapabilityProfile,
        model: Any = None,
        tokenizer: Any = None,
        *,
        training_config: Optional[TrainingConfig] = None,
        training_data_sample_size: int = 1000,
        bootstrap_iterations: int = 50,
        max_seq_length: int = 512,
        device: str = "cuda",
    ) -> None:
        self.profile = profile
        self.model = model
        self.tokenizer = tokenizer
        self.training_config = training_config or TrainingConfig()
        self.sample_size = training_data_sample_size
        self.bootstrap_iters = bootstrap_iterations
        self.max_seq_length = max_seq_length
        self.device = device

    def predict(self, training_data: Any) -> RiskReport:
        """Run prediction and return a :class:`RiskReport`."""
        t0 = time.time()
        logger.info("Predicting regression risk ...")

        if self.model is not None and self.tokenizer is not None:
            return self._predict_with_gradients(training_data, t0)
        else:
            return self._predict_from_profile_only(training_data, t0)

    # ------------------------------------------------------------------
    # gradient-based prediction (accurate — needs model + GPU)
    # ------------------------------------------------------------------

    def _predict_with_gradients(self, training_data: Any, t0: float) -> RiskReport:
        """Collect training data gradients, compute overlap with each capability subspace."""
        # Sample training data
        n_total = len(training_data) if hasattr(training_data, "__len__") else None
        sample_indices = self._sample_indices(n_total)

        # Collect gradients
        gradients: list[Tensor] = []
        texts: list[str] = []
        self.model.eval()

        for idx in sample_indices:
            example = training_data[idx]
            self.model.zero_grad()

            inputs = self._tokenize(example)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.enable_grad():
                outputs = self.model(**inputs)
                loss = outputs.loss
                loss.backward()

            grad = extract_lora_gradient(self.model)
            gradients.append(grad)
            text = example.get("text", example.get("content", ""))
            texts.append(str(text)[:200])

        G_train = torch.stack(gradients).numpy()  # (n_samples, lora_dim)

        # SVD of training gradients
        rank = min(64, G_train.shape[0] - 1, G_train.shape[1])
        _, S_train, Vt_train = randomized_svd(G_train, rank=rank)

        # Truncate to effective rank (drop noise directions)
        eff = int((S_train > S_train[0] * 1e-6).sum())
        if eff > 0:
            Vt_train = Vt_train[:eff]

        # Per-capability risk
        cap_risks: Dict[str, CapabilityRisk] = {}
        suggested_protections: Dict[str, float] = {}

        for cap_name, sub in self.profile.subspaces.items():
            overlap = subspace_overlap(sub.basis_vectors, Vt_train)

            # Per-example risk contributions
            # Project each training gradient onto capability subspace
            projections = G_train @ sub.basis_vectors.T  # (n, k)
            per_example_risk = np.linalg.norm(projections, axis=1)
            per_example_risk /= per_example_risk.sum() + 1e-12

            # Top contributing examples
            top_ids = np.argsort(per_example_risk)[::-1][:5]
            contributing = [
                ExampleRisk(
                    example_id=int(sample_indices[idx]),
                    example_text=texts[idx],
                    risk_contribution=float(per_example_risk[idx]),
                    affected_capabilities=[cap_name],
                )
                for idx in top_ids
            ]

            # Bootstrap CI for predicted delta
            predicted_delta = _overlap_to_predicted_delta(
                overlap,
                self.training_config.learning_rate,
                self.training_config.num_epochs,
                sub.baseline_loss,
            )

            bootstrap_deltas = []
            rng = np.random.RandomState(42)
            for _ in range(self.bootstrap_iters):
                boot_idx = rng.choice(len(gradients), size=len(gradients), replace=True)
                G_boot = G_train[boot_idx]
                _, _, Vt_boot = randomized_svd(G_boot, rank=rank)
                ov = subspace_overlap(sub.basis_vectors, Vt_boot)
                d = _overlap_to_predicted_delta(
                    ov,
                    self.training_config.learning_rate,
                    self.training_config.num_epochs,
                    sub.baseline_loss,
                )
                bootstrap_deltas.append(d)

            ci_lo = float(np.percentile(bootstrap_deltas, 2.5))
            ci_hi = float(np.percentile(bootstrap_deltas, 97.5))

            level = _risk_level(overlap)
            confidence = min(1.0, overlap * 3 + 0.4)

            cap_risks[cap_name] = CapabilityRisk(
                capability_name=cap_name,
                risk_level=level,
                risk_score=overlap,
                predicted_delta=predicted_delta,
                confidence_interval=(ci_lo, ci_hi),
                confidence=confidence,
                subspace_overlap=overlap,
                contributing_examples=contributing,
            )

            # Suggest protection strength
            if level in ("HIGH", "CRITICAL"):
                suggested_protections[cap_name] = 0.9
            elif level == "MEDIUM":
                suggested_protections[cap_name] = 0.5

        # Overall risk
        max_risk = max(cr.risk_score for cr in cap_risks.values()) if cap_risks else 0
        overall_level = _risk_level(max_risk)
        if overall_level in ("CRITICAL", "HIGH"):
            recommendation = f"PROTECT {', '.join(k for k, v in cap_risks.items() if v.risk_level in ('HIGH', 'CRITICAL'))} before training"
        elif overall_level == "MEDIUM":
            recommendation = "CAUTION — consider protection for medium-risk capabilities"
        else:
            recommendation = "GO — low regression risk"

        return RiskReport(
            capabilities=cap_risks,
            overall_risk=overall_level,
            overall_risk_score=max_risk,
            recommendation=recommendation,
            suggested_protections=suggested_protections,
            model_name=self.profile.model_name,
            training_data_size=n_total or len(gradients),
            training_config=self.training_config,
            compute_time_seconds=time.time() - t0,
        )

    # ------------------------------------------------------------------
    # profile-only prediction (fast, no GPU needed — uses pre-computed
    # training gradient subspace if provided as a numpy array)
    # ------------------------------------------------------------------

    def _predict_from_profile_only(
        self, training_data: Any, t0: float
    ) -> RiskReport:
        """Fallback when no model is provided — requires training_data to be a
        numpy gradient matrix or returns a stub report.
        """
        if isinstance(training_data, np.ndarray):
            G_train = training_data
        else:
            raise ValueError(
                "RegressionPredictor requires either (model + tokenizer) or "
                "a pre-computed gradient matrix (numpy array) as training_data."
            )

        rank = min(64, G_train.shape[0] - 1, G_train.shape[1])
        _, S_train, Vt_train = randomized_svd(G_train, rank=rank)
        eff = int((S_train > S_train[0] * 1e-6).sum())
        if eff > 0:
            Vt_train = Vt_train[:eff]

        cap_risks: Dict[str, CapabilityRisk] = {}
        suggested_protections: Dict[str, float] = {}

        for cap_name, sub in self.profile.subspaces.items():
            overlap = subspace_overlap(sub.basis_vectors, Vt_train)
            level = _risk_level(overlap)
            predicted_delta = _overlap_to_predicted_delta(
                overlap,
                self.training_config.learning_rate,
                self.training_config.num_epochs,
                sub.baseline_loss,
            )
            cap_risks[cap_name] = CapabilityRisk(
                capability_name=cap_name,
                risk_level=level,
                risk_score=overlap,
                predicted_delta=predicted_delta,
                confidence_interval=(predicted_delta * 1.3, predicted_delta * 0.7),
                confidence=0.5,
                subspace_overlap=overlap,
            )
            if level in ("HIGH", "CRITICAL"):
                suggested_protections[cap_name] = 0.9

        max_risk = max(cr.risk_score for cr in cap_risks.values()) if cap_risks else 0
        overall_level = _risk_level(max_risk)

        return RiskReport(
            capabilities=cap_risks,
            overall_risk=overall_level,
            overall_risk_score=max_risk,
            recommendation="GO" if overall_level == "NONE" else "CAUTION",
            suggested_protections=suggested_protections,
            model_name=self.profile.model_name,
            training_data_size=G_train.shape[0],
            training_config=self.training_config,
            compute_time_seconds=time.time() - t0,
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _sample_indices(self, n_total: int | None) -> list[int]:
        rng = np.random.RandomState(42)
        if n_total is None:
            return list(range(self.sample_size))
        n = min(self.sample_size, n_total)
        return sorted(rng.choice(n_total, size=n, replace=False).tolist())

    def _tokenize(self, example: dict) -> dict:
        text = example.get("text") or example.get("content") or ""
        enc = self.tokenizer(
            str(text),
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length,
        )
        enc["labels"] = enc["input_ids"].clone()
        return enc
