"""Sentinel — regression prevention for LLM fine-tuning."""

from __future__ import annotations

__version__ = "0.0.1"

from .profiler import CapabilityProfiler, CapabilityProfile, CapabilitySubspace
from .predictor import RegressionPredictor, RiskReport, TrainingConfig
from .optimizer import SentinelCallback
from .monitor import LiveMonitor, MonitorAlert
from .auditor import (
    RegressionAuditor,
    AuditReport,
    CapabilityDelta,
    AttributedExample,
    ConflictingExample,
    RemediationPlan,
)
from .surgeon import DataSurgeon, SurgeryPlan

__all__ = [
    "__version__",
    # Profiler
    "CapabilityProfiler",
    "CapabilityProfile",
    "CapabilitySubspace",
    # Predictor
    "RegressionPredictor",
    "RiskReport",
    "TrainingConfig",
    # Optimizer / Callback
    "SentinelCallback",
    # Monitor
    "LiveMonitor",
    "MonitorAlert",
    # Auditor
    "RegressionAuditor",
    "AuditReport",
    "CapabilityDelta",
    "AttributedExample",
    "ConflictingExample",
    "RemediationPlan",
    # Data Surgeon
    "DataSurgeon",
    "SurgeryPlan",
]
