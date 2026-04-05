from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .marker_vocabulary import REVERSAL_MARKERS
from .semantic_shift import BowEmbedder, Embedder, semantic_shift, try_sentence_transformers_embedder


@dataclass
class SCEEvent:
    rollout_text: str
    reversal_pos: int
    semantic_shift: float
    pre_segment: str
    correction_moment: str
    post_segment: str
    is_genuine: bool
    problem_id: str = ""
    step: int = 0
    difficulty: float = 0.0


class SCEDetector:
    """Detect Self-Correction Events (SCEs) in rollouts.

    Design bias: precision over recall.

    Default embedder:
    - Uses sentence-transformers if available.
    - Falls back to a hashed bag-of-words embedder otherwise.
    """

    def __init__(
        self,
        *,
        tau: float = 0.4,
        embedder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size_tokens: int = 100,
        context_chunks: int = 3,
        device: str = "cpu",
        embedder: Optional[Embedder] = None,
        return_non_genuine: bool = False,
    ) -> None:
        self.tau = float(tau)
        self.chunk_size = int(chunk_size_tokens)
        self.ctx = int(context_chunks)
        self.return_non_genuine = bool(return_non_genuine)

        if embedder is not None:
            self.embedder = embedder
        else:
            st = try_sentence_transformers_embedder(embedder_name=embedder_name, device=device)
            self.embedder = st if st is not None else BowEmbedder()

        self._shift_history: List[float] = []
        self._tau_dynamic: float = float(tau)

    def detect(
        self,
        *,
        rollout_text: str,
        is_correct: bool,
        problem_id: str = "",
        step: int = 0,
        difficulty: float = 0.0,
    ) -> Optional[SCEEvent]:
        if not is_correct:
            return None

        marker_positions = self._find_markers(rollout_text)
        if not marker_positions:
            return None

        best_event: Optional[SCEEvent] = None
        best_shift = -1.0

        for pos, marker in marker_positions:
            pre_text = rollout_text[:pos]
            post_text = rollout_text[pos:]
            shift = semantic_shift(
                pre_text=pre_text,
                post_text=post_text,
                embedder=self.embedder,
                chunk_size_tokens=self.chunk_size,
                context_chunks=self.ctx,
            )

            is_genuine = shift > self._tau_dynamic
            if is_genuine and shift > best_shift:
                correction_ctx = rollout_text[pos : pos + 200]
                best_shift = shift
                best_event = SCEEvent(
                    rollout_text=rollout_text,
                    reversal_pos=pos,
                    semantic_shift=float(shift),
                    pre_segment=pre_text,
                    correction_moment=correction_ctx,
                    post_segment=post_text,
                    is_genuine=True,
                    problem_id=problem_id,
                    step=step,
                    difficulty=float(difficulty),
                )
            elif self.return_non_genuine and best_event is None:
                # Keep the first non-genuine marker as a fallback for debugging.
                correction_ctx = rollout_text[pos : pos + 200]
                best_event = SCEEvent(
                    rollout_text=rollout_text,
                    reversal_pos=pos,
                    semantic_shift=float(shift),
                    pre_segment=pre_text,
                    correction_moment=correction_ctx,
                    post_segment=post_text,
                    is_genuine=False,
                    problem_id=problem_id,
                    step=step,
                    difficulty=float(difficulty),
                )

        if best_event is not None and best_event.is_genuine:
            self._update_dynamic_tau(best_event.semantic_shift)
            return best_event

        if best_event is not None and self.return_non_genuine:
            return best_event

        return None

    def _find_markers(self, text: str) -> List[Tuple[int, str]]:
        text_lower = text.lower()
        results: List[Tuple[int, str]] = []

        for marker in REVERSAL_MARKERS:
            m = marker.lower()
            start = 0
            while True:
                idx = text_lower.find(m, start)
                if idx == -1:
                    break
                if self._is_token_boundary(text_lower, idx, len(m)):
                    results.append((idx, marker))
                start = idx + 1

        results.sort(key=lambda x: x[0])
        return results

    @staticmethod
    def _is_token_boundary(text_lower: str, idx: int, length: int) -> bool:
        before_ok = idx == 0 or not text_lower[idx - 1].isalnum()
        after_idx = idx + length
        after_ok = after_idx >= len(text_lower) or not text_lower[after_idx].isalnum()
        return before_ok and after_ok

    def _update_dynamic_tau(self, new_shift: float) -> None:
        self._shift_history.append(float(new_shift))
        if len(self._shift_history) > 100:
            self._shift_history.pop(0)
        mean_shift = float(np.mean(self._shift_history)) if self._shift_history else self.tau
        # If mean shift drops below ~0.5, tighten threshold to resist performative drift.
        self._tau_dynamic = max(self.tau, mean_shift * 0.8)
