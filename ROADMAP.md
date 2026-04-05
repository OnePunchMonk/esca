# ESCA roadmap (starter)

This is a lightweight, repo-native version of the roadmap described in `docs/architecture_extracted.md`.

## Month 1 (Week-by-week)

- Week 1: Baseline instrumentation + calibrate τ (semantic shift threshold)
- Week 2: ESCA v0 loop (detector → replay buffer → replay step) on one domain
- Week 3: Ablations (replay-only vs reward-only, moment-only vs full-path)
- Week 4: Decisive OOD transfer experiment + falsification tree

## Month 2

- VLM extension (Qwen2.5-VL-7B + MathVista/ChartQA) + VLM marker extensions

## Month 3

- PostTrace integration hooks
- Axolotl plugin (YAML-driven)
- Docs + PyPI release for `esca-core`
- CI automation and benchmark harness skeleton

## Near-term repo work (what we’re actively building)

- `esca-core` MVP: stable API + tests + optional integrations
- Minimal utilities to run detection and export SCE traces as JSONL / HF Dataset
