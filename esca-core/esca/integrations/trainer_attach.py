from __future__ import annotations

from typing import Any


def attach_esca_callback(trainer: Any, esca_callback: Any) -> Any:
    """Attach an ESCA callback to a Transformers/TRL trainer.

    Works by duck-typing:
    - If `trainer.add_callback` exists, it is used.
    - Else, if `trainer.callback_handler.add_callback` exists, it is used.

    If `esca_callback` exposes `as_transformers_callback()`, it is called to
    produce a `transformers.TrainerCallback` instance.

    Returns the object that was attached.
    """

    cb = esca_callback.as_transformers_callback() if hasattr(esca_callback, "as_transformers_callback") else esca_callback

    if hasattr(trainer, "add_callback") and callable(getattr(trainer, "add_callback")):
        trainer.add_callback(cb)
        return cb

    callback_handler = getattr(trainer, "callback_handler", None)
    if callback_handler is not None and hasattr(callback_handler, "add_callback") and callable(getattr(callback_handler, "add_callback")):
        callback_handler.add_callback(cb)
        return cb

    raise TypeError(
        "Trainer does not expose add_callback() nor callback_handler.add_callback(); "
        "cannot attach ESCA callback."
    )
