from __future__ import annotations

from typing import Any, Dict, Iterable, List

from ..detection.sce_detector import SCEEvent


def sce_events_to_records(events: Iterable[SCEEvent]) -> List[Dict[str, Any]]:
    return [
        {
            "pre_segment": e.pre_segment,
            "correction_moment": e.correction_moment,
            "post_segment": e.post_segment,
            "semantic_shift": float(e.semantic_shift),
            "problem_id": e.problem_id,
            "step": int(e.step),
            "difficulty": float(e.difficulty),
        }
        for e in events
    ]


def build_sce_dataset(events: Iterable[SCEEvent]):
    """Create a HF `datasets.Dataset` from SCE events.

    Lazy-imports `datasets` so `esca-core` remains lightweight.
    """

    records = sce_events_to_records(events)
    try:
        from datasets import Dataset  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "HuggingFace `datasets` is not installed. Install with `pip install esca-core[train]` "
            "or `pip install datasets`."
        ) from e

    return Dataset.from_list(records)
