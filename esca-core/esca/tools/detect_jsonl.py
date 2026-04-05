from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

from ..detection.sce_detector import SCEDetector


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run(
    *,
    input_path: Path,
    output_path: Path,
    tau: float,
    reward_threshold: float,
    device: str,
    limit: Optional[int] = None,
) -> Dict[str, int]:
    detector = SCEDetector(tau=tau, device=device)

    out_records = []
    n_in = 0
    n_sce = 0

    for row in _iter_jsonl(input_path):
        n_in += 1
        if limit is not None and n_in > limit:
            break

        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            continue

        reward = float(row.get("reward", 0.0))
        is_correct = reward > reward_threshold

        event = detector.detect(
            rollout_text=text,
            is_correct=is_correct,
            problem_id=str(row.get("problem_id", "")),
            step=int(row.get("step", 0)),
            difficulty=float(row.get("difficulty", 0.0)),
        )

        if event is None:
            continue

        n_sce += 1
        out_records.append(
            {
                "problem_id": event.problem_id,
                "step": event.step,
                "difficulty": event.difficulty,
                "semantic_shift": event.semantic_shift,
                "reversal_pos": event.reversal_pos,
                "pre_segment": event.pre_segment,
                "correction_moment": event.correction_moment,
                "post_segment": event.post_segment,
            }
        )

    _write_jsonl(output_path, out_records)
    return {"inputs": n_in, "sce_events": n_sce, "written": len(out_records)}


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Detect ESCA SCE events from a JSONL of rollouts")
    p.add_argument("--in", dest="input_path", required=True, help="Input JSONL with at least {text, reward}")
    p.add_argument("--out", dest="output_path", required=True, help="Output JSONL of detected SCE events")
    p.add_argument("--tau", type=float, default=0.4, help="Semantic shift threshold")
    p.add_argument("--reward-threshold", type=float, default=0.5, help="Correctness threshold")
    p.add_argument("--device", type=str, default="cpu", help="Embedder device (if sentence-transformers installed)")
    p.add_argument("--limit", type=int, default=None, help="Optional max input rows")

    args = p.parse_args(argv)
    stats = run(
        input_path=Path(args.input_path),
        output_path=Path(args.output_path),
        tau=args.tau,
        reward_threshold=args.reward_threshold,
        device=args.device,
        limit=args.limit,
    )

    print(json.dumps(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
