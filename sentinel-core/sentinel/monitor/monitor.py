"""LiveMonitor — lightweight capability probes during training."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Literal, Optional

import numpy as np
import torch
from torch import Tensor

from ..profiler.profile import CapabilityProfile
from ..utils.lora_utils import extract_lora_params

logger = logging.getLogger("sentinel.monitor")


@dataclass
class MonitorAlert:
    timestamp: str
    step: int
    severity: Literal["INFO", "WARNING", "CRITICAL", "EMERGENCY"]
    capability: str
    metric: str
    current_value: float
    baseline_value: float
    delta: float
    message: str
    recommendation: str


class LiveMonitor:
    """Track capability health during training via lightweight probes.

    Usage::

        monitor = LiveMonitor(profile)
        # Called periodically during training:
        alerts = monitor.probe(model, step=100)
    """

    def __init__(
        self,
        profile: CapabilityProfile,
        *,
        alert_threshold: float = 0.03,
        critical_threshold: float = 0.05,
        early_stop_threshold: float = 0.10,
        trend_window: int = 5,
        alert_callback: Optional[Callable[[MonitorAlert], None]] = None,
    ) -> None:
        self.profile = profile
        self.alert_threshold = alert_threshold
        self.critical_threshold = critical_threshold
        self.early_stop_threshold = early_stop_threshold
        self.trend_window = trend_window
        self.alert_callback = alert_callback

        self._original_params: Optional[Tensor] = None
        self._history: Dict[str, list[float]] = {
            name: [] for name in profile.subspaces
        }
        self._alerts: list[MonitorAlert] = []

    def set_baseline(self, model: Any) -> None:
        """Capture the original LoRA parameters as baseline."""
        self._original_params = extract_lora_params(model)

    def probe(self, model: Any, step: int) -> List[MonitorAlert]:
        """Run a representational drift probe and return any alerts."""
        if self._original_params is None:
            self.set_baseline(model)
            return []

        current = extract_lora_params(model)
        delta = (current - self._original_params).numpy()
        delta_norm = float(np.linalg.norm(delta))

        alerts: list[MonitorAlert] = []

        for cap_name, sub in self.profile.subspaces.items():
            proj = sub.basis_vectors @ delta
            drift = float(np.linalg.norm(proj))
            drift_frac = drift / (delta_norm + 1e-8)

            self._history[cap_name].append(drift_frac)

            # Determine severity
            if drift_frac >= self.early_stop_threshold:
                severity = "EMERGENCY"
                rec = f"Auto early-stop recommended — {cap_name} drift {drift_frac:.1%}"
            elif drift_frac >= self.critical_threshold:
                severity = "CRITICAL"
                rec = f"Increase protection for {cap_name} or consider stopping"
            elif drift_frac >= self.alert_threshold:
                severity = "WARNING"
                rec = f"Monitor {cap_name} closely"
            else:
                severity = "INFO"
                rec = ""

            alert = MonitorAlert(
                timestamp=datetime.utcnow().isoformat(),
                step=step,
                severity=severity,
                capability=cap_name,
                metric="drift_fraction",
                current_value=drift_frac,
                baseline_value=0.0,
                delta=drift_frac,
                message=f"{cap_name} drift: {drift_frac:.3f} at step {step}",
                recommendation=rec,
            )
            alerts.append(alert)

            if severity != "INFO":
                logger.warning(alert.message)
                if self.alert_callback:
                    self.alert_callback(alert)

        self._alerts.extend(alerts)
        return alerts

    @property
    def all_alerts(self) -> List[MonitorAlert]:
        return list(self._alerts)

    def should_stop(self) -> bool:
        """Return True if any capability has breached the emergency threshold."""
        return any(a.severity == "EMERGENCY" for a in self._alerts)
