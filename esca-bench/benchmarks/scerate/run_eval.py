from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from esca.detection.sce_detector import SCEDetector


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def evaluate(
    *,
    input_path: Path,
    tau: float,
    reward_threshold: float,
    device: str,
    limit: Optional[int],
) -> Dict[str, Any]:
    detector = SCEDetector(tau=tau, device=device)

    n_total = 0
    n_correct = 0
    n_sce = 0

    for row in _iter_jsonl(input_path):
        n_total += 1
        if limit is not None and n_total > limit:
            break

        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            continue

        reward = float(row.get("reward", 0.0))
        is_correct = reward > reward_threshold
        if not is_correct:
            continue

        n_correct += 1
        event = detector.detect(
            rollout_text=text,
            is_correct=True,
            problem_id=str(row.get("problem_id", "")),
            step=int(row.get("step", 0)),
            difficulty=float(row.get("difficulty", 0.0)),
        )
        if event is not None:
            n_sce += 1

    sce_rate = (n_sce / n_correct) if n_correct else 0.0
    sce_per_1k_correct = sce_rate * 1000.0

    return {
        "inputs": n_total,
        "correct": n_correct,
        "sce_events": n_sce,
        "sce_rate": sce_rate,
        "sce_per_1k_correct": sce_per_1k_correct,
        "tau": tau,
        "reward_threshold": reward_threshold,
    }


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="SCERate eval: detect SCEs in correct rollouts")
    p.add_argument("--in", dest="input_path", required=True, help="Input JSONL with {text, reward}")
    p.add_argument("--tau", type=float, default=0.4)
    p.add_argument("--reward-threshold", type=float, default=0.5)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--limit", type=int, default=None)

    args = p.parse_args(argv)
    metrics = evaluate(
        input_path=Path(args.input_path),
        tau=args.tau,
        reward_threshold=args.reward_threshold,
        device=args.device,
        limit=args.limit,
    )

    print(json.dumps(metrics, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
