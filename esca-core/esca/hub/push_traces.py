from __future__ import annotations

from typing import Iterable, Optional

from ..detection.sce_detector import SCEEvent
from .dataset_builder import build_sce_dataset


def push_sce_records_to_hub(
    events: Iterable[SCEEvent],
    *,
    repo_id: str,
    private: bool = False,
    token: Optional[str] = None,
) -> None:
    """Push SCE traces to the HuggingFace Hub.

    Requires `datasets` and a working HF auth token (env var or passed in).
    """

    ds = build_sce_dataset(events)
    ds.push_to_hub(repo_id, private=private, token=token)
