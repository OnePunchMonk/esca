from __future__ import annotations

from esca.training.esca_callback import SelfCorrectionCallback
from esca.integrations.step_hook import install_training_step_hook, uninstall_training_step_hook
from esca.detection.semantic_shift import BowEmbedder


class _Trainer:
    def __init__(self) -> None:
        self.last_rollouts = []
        self.last_rewards = []

    def training_step(self, *args, **kwargs):
        # pretend the trainer just computed rollouts/rewards
        self.last_rollouts = [
            {"text": "noise. wait, I made an error. answer 42", "problem_id": "p1", "difficulty": 1.0}
        ]
        self.last_rewards = [1.0]
        return 123


def test_install_training_step_hook_feeds_esca_buffer() -> None:
    trainer = _Trainer()

    esca_cb = SelfCorrectionCallback(tau=0.0, n_replay=0)
    # keep test deterministic/lightweight
    esca_cb.detector.embedder = BowEmbedder(dim=512)

    installed = install_training_step_hook(trainer, esca_cb)
    try:
        out = trainer.training_step()
        assert out == 123
        stats = esca_cb.buffer.stats()
        assert stats["total"] >= 1
    finally:
        uninstall_training_step_hook(installed)


def test_uninstall_restores_original() -> None:
    trainer = _Trainer()
    original_func = trainer.training_step.__func__

    esca_cb = SelfCorrectionCallback(tau=0.0, n_replay=0)
    installed = install_training_step_hook(trainer, esca_cb)
    uninstall_training_step_hook(installed)

    assert "training_step" not in trainer.__dict__
    assert trainer.training_step.__func__ is original_func
