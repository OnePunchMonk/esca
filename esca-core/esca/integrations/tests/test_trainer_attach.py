from __future__ import annotations

import pytest

from esca.integrations.trainer_attach import attach_esca_callback


class _TrainerAdd:
    def __init__(self) -> None:
        self.added = []

    def add_callback(self, cb):
        self.added.append(cb)


class _Handler:
    def __init__(self) -> None:
        self.added = []

    def add_callback(self, cb):
        self.added.append(cb)


class _TrainerHandler:
    def __init__(self) -> None:
        self.callback_handler = _Handler()


class _Esca:
    def __init__(self) -> None:
        self.delegate = object()

    def as_transformers_callback(self):
        return self.delegate


def test_attach_uses_trainer_add_callback() -> None:
    t = _TrainerAdd()
    e = _Esca()
    cb = attach_esca_callback(t, e)
    assert cb is e.delegate
    assert t.added == [e.delegate]


def test_attach_uses_callback_handler() -> None:
    t = _TrainerHandler()
    e = _Esca()
    cb = attach_esca_callback(t, e)
    assert cb is e.delegate
    assert t.callback_handler.added == [e.delegate]


def test_attach_errors_when_no_hooks() -> None:
    with pytest.raises(TypeError):
        attach_esca_callback(object(), _Esca())
