"""Data structures for regression prediction results."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple


@dataclass
class TrainingConfig:
    """Training hyper-parameters needed to calibrate predictions."""

    learning_rate: float = 2e-5
    num_epochs: int = 3
    lora_r: int = 16
    lora_alpha: int = 32
    batch_size: int = 8
    optimizer: str = "adamw"
    weight_decay: float = 0.01


@dataclass
class ExampleRisk:
    """Risk contribution of a single training example."""

    example_id: int
    example_text: str  # first 200 chars
    risk_contribution: float
    affected_capabilities: List[str]


@dataclass
class CapabilityRisk:
    """Predicted regression for a single capability."""

    capability_name: str
    risk_level: Literal["NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
    risk_score: float  # 0–1, raw subspace overlap
    predicted_delta: float  # negative = regression
    confidence_interval: Tuple[float, float]
    confidence: float
    subspace_overlap: float
    contributing_examples: List[ExampleRisk] = field(default_factory=list)


@dataclass
class RiskReport:
    """Complete pre-training risk assessment."""

    capabilities: Dict[str, CapabilityRisk]
    overall_risk: Literal["NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
    overall_risk_score: float
    recommendation: str  # GO / CAUTION / STOP
    suggested_protections: Dict[str, float]  # capability → suggested β

    # Metadata
    model_name: str = ""
    training_data_size: int = 0
    training_config: Optional[TrainingConfig] = None
    compute_time_seconds: float = 0.0

    def to_json(self, path: str) -> None:
        out: dict = {
            "overall_risk": self.overall_risk,
            "overall_risk_score": self.overall_risk_score,
            "recommendation": self.recommendation,
            "suggested_protections": self.suggested_protections,
            "model_name": self.model_name,
            "training_data_size": self.training_data_size,
            "capabilities": {},
        }
        for name, cr in self.capabilities.items():
            out["capabilities"][name] = {
                "risk_level": cr.risk_level,
                "risk_score": cr.risk_score,
                "predicted_delta": cr.predicted_delta,
                "confidence_interval": list(cr.confidence_interval),
                "confidence": cr.confidence,
                "subspace_overlap": cr.subspace_overlap,
            }
        with open(path, "w") as f:
            json.dump(out, f, indent=2)

    def __str__(self) -> str:
        bar_width = 64
        lines = [
            "╔" + "═" * bar_width + "╗",
            f"║{'SENTINEL RISK REPORT':^{bar_width}}║",
            f"║  Model: {self.model_name:<{bar_width - 10}}║",
            f"║  Training Data: {self.training_data_size} examples{' ' * (bar_width - 28 - len(str(self.training_data_size)))}║",
            "╠" + "═" * bar_width + "╣",
            f"║{'':^{bar_width}}║",
            f"║  {'Capability':<18s}{'Risk':<10s}{'Predicted Δ':<16s}{'CI (95%)':<20s}║",
            f"║  {'─' * 56:<{bar_width - 2}}║",
        ]
        for name, cr in self.capabilities.items():
            ci_lo, ci_hi = cr.confidence_interval
            delta_str = f"{cr.predicted_delta:+.1f}%"
            ci_str = f"[{ci_lo:+.1f}%, {ci_hi:+.1f}%]"
            line = f"║  {name:<18s}{cr.risk_level:<10s}{delta_str:<16s}{ci_str:<20s}║"
            lines.append(line)
        lines.append(f"║{'':^{bar_width}}║")
        lines.append(
            f"║  RECOMMENDATION: {self.recommendation:<{bar_width - 19}}║"
        )
        lines.append("╚" + "═" * bar_width + "╝")
        return "\n".join(lines)
