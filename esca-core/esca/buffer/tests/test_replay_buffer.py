from __future__ import annotations

import numpy as np

from esca.buffer.replay_buffer import SCEReplayBuffer
from esca.detection.sce_detector import SCEEvent


def _sce(step: int, shift: float, difficulty: float) -> SCEEvent:
    return SCEEvent(
        rollout_text="x",
        reversal_pos=0,
        semantic_shift=shift,
        pre_segment="",
        correction_moment="wait",
        post_segment="",
        is_genuine=True,
        step=step,
        difficulty=difficulty,
    )


def test_buffer_expires_old_items() -> None:
    rng = np.random.default_rng(0)
    buf = SCEReplayBuffer(capacity=10, expiry_steps=3, rng=rng)

    buf.add(_sce(step=0, shift=0.9, difficulty=0.0), step=0)
    buf.tick()
    buf.tick()
    buf.tick()  # now _step == 3

    # item at step 0 is age 3 => expired (age < expiry_steps is required)
    assert buf.stats()["valid"] == 0


def test_buffer_samples_weighted_without_replacement() -> None:
    rng = np.random.default_rng(123)
    buf = SCEReplayBuffer(capacity=100, expiry_steps=1000, rng=rng)

    for i in range(30):
        buf.add(_sce(step=0, shift=0.1 + (i / 100.0), difficulty=float(i % 3)), step=0)

    sample = buf.sample(10)
    assert len(sample) == 10
    # no duplicates
    assert len({id(x) for x in sample}) == 10
