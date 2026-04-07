"""Sentinel — regression prevention for LLM fine-tuning."""

from __future__ import annotations

__version__ = "0.0.1"

from .profiler import CapabilityProfiler, CapabilityProfile, CapabilitySubspace
from .predictor import RegressionPredictor, RiskReport, TrainingConfig
from .optimizer import SentinelCallback
from .monitor import LiveMonitor, MonitorAlert

__all__ = [
    "__version__",
    "CapabilityProfiler",
    "CapabilityProfile",
    "CapabilitySubspace",
    "RegressionPredictor",
    "RiskReport",
    "TrainingConfig",
    "SentinelCallback",
    "LiveMonitor",
    "MonitorAlert",
]
