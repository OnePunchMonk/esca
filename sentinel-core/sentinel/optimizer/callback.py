"""SentinelCallback — TrainerCallback that protects capabilities during training."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from ..profiler.profile import CapabilityProfile
from ..utils.lora_utils import (
    extract_lora_gradient,
    extract_lora_params,
    get_lora_param_names,
)
from ..utils.svd_utils import joint_projection_matrix
from .projection import gradient_projection_step

logger = logging.getLogger("sentinel.optimizer")


class SentinelCallback:
    """Transformers / TRL ``TrainerCallback`` that performs gradient projection
    to protect capabilities during fine-tuning.

    Usage::

        callback = SentinelCallback(
            profile=profile,
            protect={"math": 0.9, "safety": 1.0},
        )
        trainer = SFTTrainer(model=model, ..., callbacks=[callback])
        trainer.train()
    """

    def __init__(
        self,
        profile: CapabilityProfile,
        protect: Union[Dict[str, float], List[str], None] = None,
        *,
        method: str = "gradient_projection",
        warmup_steps: int = 0,
        monitor: bool = True,
        monitor_interval: int = 50,
        early_stop_threshold: float = 0.10,
        log_to_jsonl: Optional[str] = "sentinel_log.jsonl",
        device: str = "cuda",
    ) -> None:
        self.profile = profile
        self.method = method
        self.warmup_steps = warmup_steps
        self.monitor = monitor
        self.monitor_interval = monitor_interval
        self.early_stop_threshold = early_stop_threshold
        self.log_path = log_to_jsonl
        self.device = device

        # Parse protection config
        if protect is None:
            self.betas: Dict[str, float] = {
                name: 0.8 for name in profile.subspaces
            }
        elif isinstance(protect, list):
            self.betas = {name: 0.8 for name in protect}
        else:
            self.betas = dict(protect)

        # Build protected subspaces dict
        self.protected_subspaces: Dict[str, np.ndarray] = {}
        for cap_name in self.betas:
            if cap_name in profile.subspaces:
                self.protected_subspaces[cap_name] = (
                    profile.subspaces[cap_name].basis_vectors
                )

        # Pre-compute joint basis for efficient projection
        bases = list(self.protected_subspaces.values())
        self._joint_basis = joint_projection_matrix(bases) if bases else None

        # Monitoring state
        self._original_lora_params: Optional[torch.Tensor] = None
        self._step_logs: list[dict] = []
        self._lora_param_names: list[str] = []

        logger.info(
            "SentinelCallback initialized — protecting: %s",
            ", ".join(f"{k} (β={v})" for k, v in self.betas.items()),
        )

    # ------------------------------------------------------------------
    # TrainerCallback hooks
    # ------------------------------------------------------------------

    def on_train_begin(self, args: Any, state: Any, control: Any, model: Any = None, **kwargs: Any) -> None:
        if model is not None:
            self._lora_param_names = get_lora_param_names(model)
            self._original_lora_params = extract_lora_params(model)
        logger.info("Protection active: %s", self.method)

    def on_step_end(self, args: Any, state: Any, control: Any, model: Any = None, **kwargs: Any) -> None:
        """Called after each optimizer step. We intercept gradients here."""
        if model is None:
            return

        step = state.global_step if hasattr(state, "global_step") else 0

        # Skip during warmup
        if step < self.warmup_steps:
            return

        # Apply gradient projection
        if self.method == "gradient_projection":
            self._apply_gradient_projection(model)

        # Monitor at intervals
        if self.monitor and step > 0 and step % self.monitor_interval == 0:
            drift = self._compute_drift(model)
            self._step_logs.append({"step": step, "drift": drift})
            for cap_name, d in drift.items():
                if d["drift_fraction"] > self.early_stop_threshold:
                    logger.warning(
                        "Step %d: %s drift=%.3f exceeds threshold %.3f — consider early stopping",
                        step,
                        cap_name,
                        d["drift_fraction"],
                        self.early_stop_threshold,
                    )

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        logger.info(
            "Training complete. %d monitoring checkpoints recorded.",
            len(self._step_logs),
        )
        if self.log_path:
            self._write_logs()

    # ------------------------------------------------------------------
    # The actual projection (called before optimizer.step in practice,
    # or we can modify gradients in on_step_end for the next step).
    #
    # NOTE: In a production implementation, this would hook into
    # `on_pre_optimizer_step` or a custom gradient hook.  For the first
    # iteration we modify the gradient *in-place* on the model's params.
    # ------------------------------------------------------------------

    def _apply_gradient_projection(self, model: Any) -> None:
        """Project current LoRA gradients to remove protected components."""
        grad = extract_lora_gradient(model)
        if grad.abs().sum() == 0:
            return

        grad_protected = gradient_projection_step(
            grad,
            self.protected_subspaces,
            self.betas,
            joint_basis=self._joint_basis,
        )

        # Write the projected gradient back
        offset = 0
        for name, p in model.named_parameters():
            if not p.requires_grad or "lora_" not in name:
                continue
            numel = p.numel()
            if p.grad is not None:
                p.grad.copy_(
                    grad_protected[offset : offset + numel]
                    .reshape(p.grad.shape)
                    .to(p.grad.device, p.grad.dtype)
                )
            offset += numel

    def _compute_drift(self, model: Any) -> Dict[str, Dict[str, float]]:
        """Measure how much LoRA params have drifted into each capability subspace."""
        if self._original_lora_params is None:
            return {}

        current = extract_lora_params(model)
        delta = (current - self._original_lora_params).numpy()
        delta_norm = float(np.linalg.norm(delta))

        drift: Dict[str, Dict[str, float]] = {}
        for cap_name, basis in self.protected_subspaces.items():
            proj = basis @ delta  # (k,)
            proj_norm = float(np.linalg.norm(proj))
            drift[cap_name] = {
                "drift_magnitude": proj_norm,
                "drift_fraction": proj_norm / (delta_norm + 1e-8),
                "total_param_change": delta_norm,
            }
        return drift

    def _write_logs(self) -> None:
        import json

        with open(self.log_path, "w") as f:
            for entry in self._step_logs:
                f.write(json.dumps(entry, default=str) + "\n")
