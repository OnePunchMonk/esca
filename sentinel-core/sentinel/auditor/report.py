"""Data classes for post-training audit results."""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


@dataclass
class AttributedExample:
    """A training example with its causal influence on a capability."""

    example_id: int
    example_text: str  # First 500 chars
    influence_score: float  # Positive = helpful, negative = harmful
    gradient_similarity: float  # Raw cosine similarity in LoRA subspace
    confidence: float = 1.0  # Estimation confidence

    # Cross-capability effects
    effects_on_other_capabilities: Dict[str, float] = field(default_factory=dict)
    # e.g. {"math": -0.15, "customer_support": +0.3}


@dataclass
class CapabilityDelta:
    """What changed for a single capability after training."""

    capability_name: str

    # Loss metrics (always available — no eval set needed)
    baseline_loss: float
    final_loss: float
    loss_delta: float  # final - baseline (positive = regression)

    # Attribution
    top_helpful_examples: List[AttributedExample] = field(default_factory=list)
    top_harmful_examples: List[AttributedExample] = field(default_factory=list)

    # Representational drift
    drift_magnitude: float = 0.0  # L2 norm of param change in subspace
    drift_fraction: float = 0.0  # Relative fraction of total param change

    # Prediction comparison (populated when risk report was run beforehand)
    predicted_delta: Optional[float] = None
    prediction_error: Optional[float] = None  # actual - predicted

    @property
    def regressed(self) -> bool:
        return self.loss_delta > 0

    @property
    def severity(self) -> str:
        if self.loss_delta > 0.5:
            return "CRITICAL"
        elif self.loss_delta > 0.2:
            return "HIGH"
        elif self.loss_delta > 0.05:
            return "MEDIUM"
        elif self.loss_delta > 0:
            return "LOW"
        return "NONE"


@dataclass
class ConflictingExample:
    """A training example with opposing effects on different capabilities."""

    example_id: int
    example_text: str
    positive_effects: Dict[str, float]  # capability → positive influence
    negative_effects: Dict[str, float]  # capability → negative influence
    net_value: float
    recommendation: Literal["KEEP", "REMOVE", "REWEIGHT", "REVIEW"] = "REVIEW"


@dataclass
class RemediationPlan:
    """Concrete, actionable steps to fix detected regressions."""

    # Option 1: Remove harmful examples
    examples_to_remove: List[int] = field(default_factory=list)
    expected_regression_recovery: Dict[str, float] = field(default_factory=dict)
    expected_target_task_cost: float = 0.0

    # Option 2: Reweight examples
    example_weights: Dict[int, float] = field(default_factory=dict)

    # Option 3: Add retention data
    retention_data_recommendations: Dict[str, int] = field(default_factory=dict)
    retention_data_sources: Dict[str, str] = field(default_factory=dict)

    # Option 4: Adjust training config
    suggested_config_changes: Dict[str, Any] = field(default_factory=dict)

    recommendation_summary: str = ""

    def summary(self) -> str:
        lines = ["  Remediation Plan:"]
        if self.examples_to_remove:
            lines.append(
                f"    • Remove {len(self.examples_to_remove)} harmful examples"
                f" (expected target-cost: {self.expected_target_task_cost:.1%})"
            )
        if self.retention_data_recommendations:
            for cap, n in self.retention_data_recommendations.items():
                src = self.retention_data_sources.get(cap, "sentinel:standard")
                lines.append(f"    • Add {n} retention examples for '{cap}' ({src})")
        if self.suggested_config_changes:
            changes = ", ".join(f"{k}={v}" for k, v in self.suggested_config_changes.items())
            lines.append(f"    • Suggested config: {changes}")
        if self.recommendation_summary:
            lines.append(f"  → {self.recommendation_summary}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Top-level report
# ---------------------------------------------------------------------------


@dataclass
class AuditReport:
    """Complete post-training regression audit."""

    # Per-capability deltas
    capability_deltas: Dict[str, CapabilityDelta]

    # Classification
    capabilities_improved: List[str] = field(default_factory=list)
    capabilities_degraded: List[str] = field(default_factory=list)
    capabilities_unchanged: List[str] = field(default_factory=list)

    overall_regression_score: float = 0.0  # 0 = no regression, 1 = catastrophic

    conflicting_examples: List[ConflictingExample] = field(default_factory=list)
    remediation: Optional[RemediationPlan] = None

    # Metadata
    model_name: str = ""
    training_data_size: int = 0
    attribution_method: str = "gradient_cosine"
    compute_time_seconds: float = 0.0

    # ---------------------------------------------------------------------------
    # Output methods
    # ---------------------------------------------------------------------------

    def summary(self) -> str:
        lines = [
            "╔══════════════════════════════════════════════════════════════╗",
            "║              SENTINEL AUDIT REPORT                          ║",
            f"║  Model: {self.model_name:<52} ║",
            f"║  Training data: {self.training_data_size:<44} ║",
            f"║  Attribution: {self.attribution_method:<46} ║",
            "╠══════════════════════════════════════════════════════════════╣",
            "║                                                              ║",
            "║  Capability      Loss Δ      Severity    Drift               ║",
            "║  ──────────────────────────────────────────────             ║",
        ]
        for name, delta in sorted(self.capability_deltas.items()):
            sev = delta.severity
            lines.append(
                f"║  {name:<14}{delta.loss_delta:+.4f}      {sev:<12}{delta.drift_fraction:.3f}          ║"
            )
        lines += [
            "║                                                              ║",
            f"║  Overall regression score: {self.overall_regression_score:.3f}                        ║",
        ]
        if self.capabilities_degraded:
            caps = ", ".join(self.capabilities_degraded)
            lines.append(f"║  Degraded: {caps:<50} ║")
        if self.capabilities_improved:
            caps = ", ".join(self.capabilities_improved)
            lines.append(f"║  Improved: {caps:<50} ║")
        lines.append(
            "╚══════════════════════════════════════════════════════════════╝"
        )
        if self.remediation:
            lines.append(self.remediation.summary())
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    def to_json(self, path: str) -> None:
        """Write a JSON-serialisable snapshot of the report to *path*."""
        import dataclasses

        def _default(obj: Any) -> Any:
            if hasattr(obj, "__dataclass_fields__"):
                return dataclasses.asdict(obj)
            return str(obj)

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "model_name": self.model_name,
                    "training_data_size": self.training_data_size,
                    "attribution_method": self.attribution_method,
                    "overall_regression_score": self.overall_regression_score,
                    "capabilities_improved": self.capabilities_improved,
                    "capabilities_degraded": self.capabilities_degraded,
                    "capabilities_unchanged": self.capabilities_unchanged,
                    "capability_deltas": {
                        name: {
                            "baseline_loss": d.baseline_loss,
                            "final_loss": d.final_loss,
                            "loss_delta": d.loss_delta,
                            "severity": d.severity,
                            "drift_magnitude": d.drift_magnitude,
                            "drift_fraction": d.drift_fraction,
                            "predicted_delta": d.predicted_delta,
                            "prediction_error": d.prediction_error,
                            "top_harmful": [
                                {
                                    "example_id": e.example_id,
                                    "example_text": e.example_text,
                                    "influence_score": e.influence_score,
                                }
                                for e in d.top_harmful_examples
                            ],
                            "top_helpful": [
                                {
                                    "example_id": e.example_id,
                                    "example_text": e.example_text,
                                    "influence_score": e.influence_score,
                                }
                                for e in d.top_helpful_examples
                            ],
                        }
                        for name, d in self.capability_deltas.items()
                    },
                    "compute_time_seconds": self.compute_time_seconds,
                },
                fh,
                indent=2,
                default=_default,
            )

    def to_html(self, path: str) -> None:
        """Write a self-contained HTML audit report to *path*."""
        html = _build_html_report(self)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html)


# ---------------------------------------------------------------------------
# HTML report builder
# ---------------------------------------------------------------------------


def _build_html_report(report: AuditReport) -> str:
    """Build a minimal but readable HTML audit report."""

    def _sev_color(sev: str) -> str:
        return {
            "CRITICAL": "#ef4444",
            "HIGH": "#f97316",
            "MEDIUM": "#eab308",
            "LOW": "#84cc16",
            "NONE": "#22c55e",
        }.get(sev, "#6b7280")

    rows = []
    for name, d in sorted(report.capability_deltas.items()):
        color = _sev_color(d.severity)
        harm_list = "".join(
            f"<li>#{e.example_id}: {e.example_text[:80]}… "
            f"(influence={e.influence_score:+.3f})</li>"
            for e in d.top_harmful_examples[:3]
        )
        rows.append(
            f"""
            <tr>
              <td><strong>{name}</strong></td>
              <td>{d.baseline_loss:.4f}</td>
              <td>{d.final_loss:.4f}</td>
              <td style="color:{color};font-weight:700">{d.loss_delta:+.4f}</td>
              <td style="color:{color}">{d.severity}</td>
              <td>{d.drift_fraction:.3f}</td>
              <td><ul style="margin:0;padding-left:1em">{harm_list}</ul></td>
            </tr>"""
        )

    remediation_html = ""
    if report.remediation:
        rem = report.remediation
        remediation_html = f"""
        <section>
          <h2>Remediation Recommendations</h2>
          <pre style="background:#1e1e1e;color:#d4d4d4;padding:1em;border-radius:6px">{rem.summary()}</pre>
        </section>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Sentinel Audit Report — {report.model_name}</title>
  <style>
    body {{font-family:system-ui,sans-serif;background:#0f0f0f;color:#e5e5e5;margin:0;padding:2rem}}
    h1 {{color:#a78bfa;font-size:1.8rem;margin-bottom:0.25rem}}
    h2 {{color:#818cf8;font-size:1.2rem;margin-top:2rem}}
    .meta {{color:#6b7280;font-size:0.85rem;margin-bottom:2rem}}
    .score {{display:inline-block;padding:0.4em 1em;border-radius:99px;
             background:#1e1e2e;border:2px solid #a78bfa;font-size:1.1rem;font-weight:700}}
    table {{width:100%;border-collapse:collapse;font-size:0.9rem;margin-top:1rem}}
    th {{text-align:left;color:#818cf8;border-bottom:1px solid #374151;padding:0.5em}}
    td {{padding:0.5em;border-bottom:1px solid #1f2937;vertical-align:top}}
    tr:hover td {{background:#1f2937}}
    section {{margin-top:2rem}}
    .chips {{display:flex;gap:0.5rem;flex-wrap:wrap;margin-top:0.5rem}}
    .chip {{padding:0.2em 0.75em;border-radius:99px;font-size:0.8rem}}
    .chip-bad {{background:#7f1d1d;color:#fca5a5}}
    .chip-good {{background:#14532d;color:#86efac}}
  </style>
</head>
<body>
  <h1>&#128737; Sentinel Audit Report</h1>
  <p class="meta">
    Model: <strong>{report.model_name}</strong> &nbsp;|&nbsp;
    Training data: <strong>{report.training_data_size:,} examples</strong> &nbsp;|&nbsp;
    Attribution: <strong>{report.attribution_method}</strong> &nbsp;|&nbsp;
    Computed in: <strong>{report.compute_time_seconds:.1f}s</strong>
  </p>

  <section>
    <h2>Overall Regression Score</h2>
    <span class="score">{report.overall_regression_score:.3f}</span>
    <div class="chips">
      {"".join(f'<span class="chip chip-bad">{c} ↓</span>' for c in report.capabilities_degraded)}
      {"".join(f'<span class="chip chip-good">{c} ↑</span>' for c in report.capabilities_improved)}
    </div>
  </section>

  <section>
    <h2>Capability Deltas</h2>
    <table>
      <thead>
        <tr>
          <th>Capability</th>
          <th>Baseline Loss</th>
          <th>Final Loss</th>
          <th>Loss Δ</th>
          <th>Severity</th>
          <th>Drift Fraction</th>
          <th>Top Harmful Examples</th>
        </tr>
      </thead>
      <tbody>{"".join(rows)}</tbody>
    </table>
  </section>

  {remediation_html}
</body>
</html>"""
