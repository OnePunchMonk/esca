# esca-core

Minimal core library for **ESCA (Emergent Self-Correction Amplification)**.

## Install (editable)

```bash
python -m pip install -e esca-core[test]
```

## Quick sanity

```bash
python -c "import esca; print(esca.__version__); print(esca.SCEDetector)"
pytest -q esca-core
```

## What’s implemented (initial)

- `esca.detection.SCEDetector` + `esca.detection.SCEEvent`
- `esca.buffer.SCEReplayBuffer`
- `esca.training.SelfCorrectionCallback` (minimal callback surface)
- `esca.integrations.attach_esca_callback` (duck-typed trainer attachment)

Heavyweight integrations (TRL/Transformers/W&B/HF Datasets) are optional and are intentionally stubbed/guarded at this stage.

## Optional: attach to a Trainer/GRPOTrainer

When using Transformers/TRL, create ESCA and attach it to your trainer:

```python
from esca import SelfCorrectionCallback, attach_esca_callback

esca_cb = SelfCorrectionCallback(tau=0.4, n_replay=50, replay_batch_size=32)
attach_esca_callback(trainer, esca_cb)
```

Note: `SelfCorrectionCallback.as_transformers_callback()` imports `transformers` lazily, only when you attach.

## Optional: feed rollouts without TRL

If you have your own loop (or want to integrate incrementally), call:

```python
esca_cb.consume_rollouts(rollouts=rollouts, rewards=rewards, model=model, optimizer=optimizer)
```

There is also a duck-typed helper that can hook a trainer’s `training_step()` and
call `consume_rollouts()` after each step:

```python
from esca.integrations import install_training_step_hook

install_training_step_hook(trainer, esca_cb)
```
