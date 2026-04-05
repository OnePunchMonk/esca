from __future__ import annotations

__all__ = [
    "__version__",
    "SCEEvent",
    "SCEDetector",
    "SCEReplayBuffer",
    "SelfCorrectionCallback",
    "REVERSAL_MARKERS",
    "sce_events_to_records",
    "build_sce_dataset",
    "push_sce_records_to_hub",
    "attach_esca_callback",
]

__version__ = "0.0.1"

from .detection.marker_vocabulary import REVERSAL_MARKERS
from .detection.sce_detector import SCEEvent, SCEDetector
from .buffer.replay_buffer import SCEReplayBuffer
from .training.esca_callback import SelfCorrectionCallback
from .hub.dataset_builder import build_sce_dataset, sce_events_to_records
from .hub.push_traces import push_sce_records_to_hub
from .integrations.trainer_attach import attach_esca_callback
