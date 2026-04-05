from __future__ import annotations

from esca.detection.sce_detector import SCEEvent
from esca.hub.dataset_builder import sce_events_to_records


def test_sce_events_to_records_shape() -> None:
    e = SCEEvent(
        rollout_text="x",
        reversal_pos=1,
        semantic_shift=0.7,
        pre_segment="a",
        correction_moment="wait",
        post_segment="b",
        is_genuine=True,
        problem_id="p",
        step=3,
        difficulty=2.0,
    )

    records = sce_events_to_records([e])
    assert len(records) == 1
    r = records[0]
    assert r["problem_id"] == "p"
    assert r["step"] == 3
    assert isinstance(r["semantic_shift"], float)
