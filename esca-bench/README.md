# esca-bench

Benchmark harness scaffold for ESCA.

## SCERate (starter)

A minimal evaluation that measures:
- `sce_rate`: fraction of *correct* rollouts containing a detected genuine SCE
- `sce_per_1k_correct`: SCEs per 1000 correct rollouts

### Input format

JSONL, one record per rollout:

```json
{"text": "...", "reward": 1.0, "problem_id": "...", "difficulty": 2.0}
```

Only `text` and `reward` are required.

### Run

From repo root:

```bash
python -m pip install -e esca-core
python esca-bench/benchmarks/scerate/run_eval.py --in esca-bench/examples/rollouts.jsonl --tau 0.4
```
