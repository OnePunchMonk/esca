from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from ..buffer.replay_buffer import SCEReplayBuffer
from ..detection.sce_detector import SCEDetector
from ..logging.diagnostics import ESCADiagnostics
from .sft_step import run_sft_on_correction_moments


class SelfCorrectionCallback:
    """Primary ESCA entry point.

    Intended to be used as a callback in training loops (Transformers / TRL).

    This initial implementation focuses on:
    - detecting SCEs from (rollouts, rewards)
    - maintaining a replay buffer
    - exposing a stable surface for future integration work
    """

    def __init__(
        self,
        *,
        tau: float = 0.4,
        n_replay: int = 50,
        replay_batch_size: int = 32,
        alpha: float = 0.0,
        buffer_capacity: int = 50_000,
        expiry_steps: int = 500,
        push_to_hub: bool = False,
        hub_repo_id: Optional[str] = None,
        hub_private: bool = False,
        hub_token: Optional[str] = None,
        wandb_log: bool = False,
        diagnostics_jsonl: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        self.detector = SCEDetector(tau=tau, device=device)
        self.buffer = SCEReplayBuffer(capacity=buffer_capacity, expiry_steps=expiry_steps)
        self.diagnostics = ESCADiagnostics(
            jsonl_path=diagnostics_jsonl,
            wandb_log=wandb_log,
        )

        self.n_replay = int(n_replay)
        self.replay_bs = int(replay_batch_size)
        self.alpha = float(alpha)

        self.push_to_hub = bool(push_to_hub)
        self.hub_repo_id = hub_repo_id
        self.hub_private = bool(hub_private)
        self.hub_token = hub_token

        self._step = 0

    def consume_rollouts(
        self,
        *,
        rollouts,
        rewards,
        model=None,
        optimizer=None,
        reward_threshold: float = 0.5,
    ) -> None:
        """Consume rollouts/rewards for one training step.

        This is the dependency-free integration surface. Framework adapters can
        call this directly after they compute rollouts/rewards.
        """

        # Phase 1: detect SCEs
        for rollout, reward in zip(rollouts or [], rewards or []):
            text = rollout.get("text") if isinstance(rollout, dict) else None
            if not text:
                continue

            is_correct = float(reward) > float(reward_threshold)
            sce = self.detector.detect(
                rollout_text=text,
                is_correct=is_correct,
                problem_id=str(rollout.get("problem_id", "")) if isinstance(rollout, dict) else "",
                step=self._step,
                difficulty=float(rollout.get("difficulty", 0.0)) if isinstance(rollout, dict) else 0.0,
            )
            if sce is not None:
                self.buffer.add(sce, self._step)

        # Phase 2: reward bonus (placeholder)
        _ = self.alpha

        # Phase 3: optional replay hook
        if self.n_replay > 0 and self._step % self.n_replay == 0:
            if self.replay_bs > 0 and self.buffer.stats().get("valid", 0) >= self.replay_bs:
                if model is not None and optimizer is not None:
                    sce_batch = self.buffer.sample(self.replay_bs)
                    run_sft_on_correction_moments(model, optimizer, sce_batch)

        # Phase 4: log
        self.diagnostics.log(self._step, self.buffer.stats(), rollouts=rollouts or [], rewards=rewards or [])

        self.buffer.tick()
        self._step += 1

    def on_step_end(self, args=None, state=None, control=None, **kwargs):
        self.consume_rollouts(
            rollouts=kwargs.get("rollouts", []),
            rewards=kwargs.get("rewards", []),
            model=kwargs.get("model"),
            optimizer=kwargs.get("optimizer"),
            reward_threshold=float(kwargs.get("reward_threshold", 0.5)),
        )
        return control

    def on_train_end(self, args=None, state=None, control=None, **kwargs):
        if not self.push_to_hub:
            return control

        if not self.hub_repo_id:
            raise ValueError("push_to_hub=True requires hub_repo_id")

        # Lazy import so `import esca` doesn't pull in datasets.
        from ..hub.push_traces import push_sce_records_to_hub

        push_sce_records_to_hub(
            list(self.buffer._buffer),
            repo_id=self.hub_repo_id,
            private=self.hub_private,
            token=self.hub_token,
        )

        return control

    def as_transformers_callback(self):
        """Return a `transformers.TrainerCallback` that delegates to this instance.

        This keeps `esca-core` import-time lightweight while still supporting
        first-class integration with Transformers/TRL when those dependencies
        are present.
        """

        try:
            from transformers import TrainerCallback  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "`transformers` is required for Trainer callback integration. "
                "Install with `pip install esca-core[train]` (or `pip install transformers`)."
            ) from e

        parent = self

        class _DelegatingCallback(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):  # type: ignore[override]
                return parent.on_step_end(args=args, state=state, control=control, **kwargs)

            def on_train_end(self, args, state, control, **kwargs):  # type: ignore[override]
                return parent.on_train_end(args=args, state=state, control=control, **kwargs)

        return _DelegatingCallback()
