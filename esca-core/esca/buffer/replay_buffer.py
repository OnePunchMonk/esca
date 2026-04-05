from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

import numpy as np

from ..detection.sce_detector import SCEEvent


@dataclass(frozen=True)
class ReplayStats:
    total: int
    valid: int
    mean_shift: float
    mean_difficulty: float
    step: int


class SCEReplayBuffer:
    """Priority replay buffer for Self-Correction Events.

    Sampling weight is the product of:
    - Recency: (1 - age/expiry_steps)
    - Shift magnitude: semantic_shift
    - Difficulty: (1 + difficulty_alpha * difficulty)
    """

    def __init__(
        self,
        *,
        capacity: int = 50_000,
        expiry_steps: int = 500,
        difficulty_alpha: float = 0.5,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        if expiry_steps <= 0:
            raise ValueError("expiry_steps must be > 0")

        self.capacity = int(capacity)
        self.expiry_steps = int(expiry_steps)
        self.difficulty_alpha = float(difficulty_alpha)
        self._buffer: Deque[SCEEvent] = deque(maxlen=self.capacity)
        self._step = 0
        self._rng = rng if rng is not None else np.random.default_rng()

    def add(self, sce: SCEEvent, step: int) -> None:
        # Copy-like update: we mutate the dataclass fields that are tracking metadata.
        sce.step = int(step)
        self._buffer.append(sce)

    def tick(self) -> None:
        self._step += 1

    def _valid(self) -> List[SCEEvent]:
        return [s for s in self._buffer if (self._step - s.step) < self.expiry_steps]

    def sample(self, n: int) -> List[SCEEvent]:
        if n <= 0:
            return []

        valid = self._valid()
        if len(valid) <= n:
            return list(valid)

        ages = np.array([self._step - s.step for s in valid], dtype=np.float32)
        recency = 1.0 - (ages / float(self.expiry_steps))
        recency = np.clip(recency, 0.0, 1.0)

        shift = np.array([s.semantic_shift for s in valid], dtype=np.float32)
        difficulty = np.array([s.difficulty for s in valid], dtype=np.float32)

        weights = shift * recency * (1.0 + self.difficulty_alpha * difficulty)
        weights = np.clip(weights, 1e-8, None)
        weights /= weights.sum()

        idxs = self._rng.choice(len(valid), size=n, replace=False, p=weights)
        return [valid[int(i)] for i in idxs]

    def stats(self) -> Dict[str, float | int]:
        valid = self._valid()
        mean_shift = float(np.mean([s.semantic_shift for s in valid])) if valid else 0.0
        mean_difficulty = float(np.mean([s.difficulty for s in valid])) if valid else 0.0
        return {
            "total": len(self._buffer),
            "valid": len(valid),
            "mean_shift": mean_shift,
            "mean_difficulty": mean_difficulty,
            "step": self._step,
        }
