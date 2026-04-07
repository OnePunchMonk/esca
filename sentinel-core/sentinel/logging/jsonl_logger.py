"""Simple JSONL logger for Sentinel metrics."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


class JSONLLogger:
    """Append-only JSONL log for training metrics."""

    def __init__(self, path: str = "sentinel_log.jsonl") -> None:
        self.path = Path(path)

    def log(self, data: Dict[str, Any]) -> None:
        data["_timestamp"] = datetime.utcnow().isoformat()
        with open(self.path, "a") as f:
            f.write(json.dumps(data, default=str) + "\n")
