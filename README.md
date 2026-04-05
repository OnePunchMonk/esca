# ESCA

**Emergent Self‑Correction Amplification** — a production-grade library + research framework for amplifying self-correction in RLVR-trained language models.

This repo currently contains:

- `esca-core/`: minimal Python package with SCE detection + replay buffer + callback surface
- `docs/architecture_extracted.md`: extracted architecture document
- `goals.md`: short milestone list

## Quickstart (dev)

Prereqs: Python 3.10+.

```bash
python -m pip install -e esca-core[test]
python -c "import esca; print(esca.__version__)"
pytest -q esca-core
```

## Repo docs

- Architecture: `docs/architecture_extracted.md`
- Goals: `goals.md`
- Roadmap: `ROADMAP.md`

## Status

Early scaffold (April 2026). `esca-core` is intentionally lightweight and keeps heavyweight dependencies optional.
