# ESCA goals

This repo is the start of **ESCA (Emergent Self‑Correction Amplification)** — a production-grade library + research framework for amplifying self-correction in RLVR-trained language models.

## Guiding principles

- **Benchmark-gated**: ideas should survive the 4‑benchmark critique before deep investment.
- **Precision over recall** (for SCE detection): false positives are expensive.
- **GitHub-first**: design for reproducibility, tests, and modular integrations.

## Milestones (initial)

### M0 — Repo bootstrap (done when)

- `goals.md` exists and captures milestones.
- Architecture text is accessible in-repo (see `docs/architecture_extracted.md`).

### M1 — `esca-core` minimal usable library

Deliver a small, testable, pip-installable core that matches the architecture sketch:

- **SCE detection**: `SCEDetector` + `SCEEvent` + reversal marker vocabulary.
- **Replay buffer**: `SCEReplayBuffer` with capacity, expiry, and weighted sampling.
- **Training hook surface**: `SelfCorrectionCallback` stub that can be wired into TRL/Transformers later (imports should be optional).
- **Diagnostics surface**: a minimal `ESCADiagnostics` logger interface.

Done when:

- `python -c "import esca; print(esca.__version__)"` works after installing `esca-core`.
- Unit tests pass locally (`pytest`), without requiring GPUs.

### M2 — Integration slices (follow-on)

- TRL GRPO integration glue (callback actually receives rollouts/rewards).
- HuggingFace Datasets push of SCE traces.
- Optional W&B logging.

### M3 — Bench harness skeleton (follow-on)

- `esca-bench` scaffold with at least one runnable benchmark script and a stable metric schema.

### M4 — Roadmap alignment (research program)

- Week-by-week Month 1 execution (τ calibration → ESCA v0 loop → ablations → decisive OOD experiment).
- Month 2 VLM extension planning.
- Month 3 polish/community: docs, Axolotl plugin, PyPI release.

## Implementation starting point

- Start with **M1**: implement `esca-core` with a minimal API and fast tests.
- Keep heavyweight deps optional; default behavior should work with a lightweight semantic-shift backend.
