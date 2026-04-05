from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple


RolloutGetter = Callable[[Any], Tuple[Any, Any]]


def _default_getter(trainer: Any) -> Tuple[Any, Any]:
    """Best-effort extraction of (rollouts, rewards) from a trainer.

    Different frameworks store these differently; we try a few common names.
    """

    for r_name in ("rollouts", "_rollouts", "last_rollouts", "_last_rollouts"):
        if hasattr(trainer, r_name):
            rollouts = getattr(trainer, r_name)
            break
    else:
        rollouts = []

    for w_name in ("rewards", "_rewards", "last_rewards", "_last_rewards"):
        if hasattr(trainer, w_name):
            rewards = getattr(trainer, w_name)
            break
    else:
        rewards = []

    return rollouts, rewards


@dataclass
class InstalledStepHook:
    trainer: Any
    original_had_instance_attr: bool
    original_training_step_attr: Any


def install_training_step_hook(
    trainer: Any,
    esca_callback: Any,
    *,
    getter: Optional[RolloutGetter] = None,
    reward_threshold: float = 0.5,
) -> InstalledStepHook:
    """Monkeypatch `trainer.training_step` to feed rollouts/rewards into ESCA.

    This is intentionally duck-typed and optional; it does not require TRL.

    Expectations:
    - `trainer.training_step(*args, **kwargs)` exists.
    - After the original training_step runs, `getter(trainer)` can return
      (rollouts, rewards), or they can be exposed on the trainer via common
      attribute names.

    The hook calls `esca_callback.consume_rollouts(...)`.
    """

    if not hasattr(trainer, "training_step") or not callable(getattr(trainer, "training_step")):
        raise TypeError("trainer must define a callable training_step")

    if not hasattr(esca_callback, "consume_rollouts") or not callable(getattr(esca_callback, "consume_rollouts")):
        raise TypeError("esca_callback must define consume_rollouts(rollouts=..., rewards=...)")

    original_had_instance_attr = "training_step" in getattr(trainer, "__dict__", {})
    original_attr = trainer.__dict__.get("training_step") if hasattr(trainer, "__dict__") else None
    original_callable = trainer.training_step
    getter_fn = getter if getter is not None else _default_getter

    def _wrapped_training_step(*args, **kwargs):
        out = original_callable(*args, **kwargs)

        rollouts, rewards = getter_fn(trainer)
        model = getattr(trainer, "model", None)
        optimizer = getattr(trainer, "optimizer", None)

        esca_callback.consume_rollouts(
            rollouts=rollouts,
            rewards=rewards,
            model=model,
            optimizer=optimizer,
            reward_threshold=reward_threshold,
        )

        return out

    trainer.training_step = _wrapped_training_step
    return InstalledStepHook(
        trainer=trainer,
        original_had_instance_attr=original_had_instance_attr,
        original_training_step_attr=original_attr,
    )


def uninstall_training_step_hook(installed: InstalledStepHook) -> None:
    if installed.original_had_instance_attr:
        installed.trainer.training_step = installed.original_training_step_attr
        return

    # Original was resolved via class descriptor; remove our instance override.
    if hasattr(installed.trainer, "__dict__") and "training_step" in installed.trainer.__dict__:
        del installed.trainer.__dict__["training_step"]
