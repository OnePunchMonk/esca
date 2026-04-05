from __future__ import annotations

from typing import Iterable, Optional

from ..detection.sce_detector import SCEEvent


def run_sft_on_correction_moments(
    model,
    optimizer,
    sce_batch: Iterable[SCEEvent],
    *,
    tokenizer=None,
    strict: bool = False,
) -> None:
    """Run a supplementary SFT step on correction-moment tokens.

    This is a placeholder surface for the architecture's Phase 3.

    At this repo stage, we keep it as a no-op by default so that:
    - `esca-core` remains installable without torch/transformers
    - integration can be built incrementally (TRL/Transformers hooks)

    Set `strict=True` to force a hard failure if you expect this to be implemented.
    """

    if strict:
        raise NotImplementedError(
            "SFT-on-correction-moments is not implemented yet; "
            "wire this into a concrete trainer loop first."
        )

    # Intentionally no-op.
    return
