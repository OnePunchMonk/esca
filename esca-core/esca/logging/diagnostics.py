from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


@dataclass
class ESCADiagnostics:
    """Minimal diagnostics logger.

    This is intentionally lightweight at this stage:
    - Writes one JSONL record per step (optional)
    - Optionally forwards metrics to W&B when installed
    """

    jsonl_path: Optional[Path] = None
    wandb_log: bool = False

    def __post_init__(self) -> None:
        if self.jsonl_path is not None:
            self.jsonl_path = Path(self.jsonl_path)

    def log(
        self,
        step: int,
        buffer_stats: Dict[str, Any],
        rollouts: Iterable[Dict[str, Any]] = (),
        rewards: Iterable[float] = (),
    ) -> None:
        record: Dict[str, Any] = {
            "step": int(step),
            "buffer": buffer_stats,
        }

        if self.jsonl_path is not None:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with self.jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if self.wandb_log:
            try:
                import wandb

                wandb.log({"esca/buffer_total": buffer_stats.get("total", 0), **record}, step=step)
            except Exception:
                # If wandb isn't configured, stay silent and non-fatal.
                pass
