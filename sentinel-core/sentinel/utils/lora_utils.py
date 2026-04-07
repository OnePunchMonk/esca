"""Utilities for extracting and manipulating LoRA parameters and gradients."""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn, Tensor


def _is_lora_param(name: str) -> bool:
    """Return True if *name* looks like a PEFT LoRA parameter."""
    return "lora_" in name and ("weight" in name or "default" in name)


def get_lora_param_names(model: nn.Module) -> List[str]:
    """Return the names of all trainable LoRA parameters in order."""
    return [
        name
        for name, p in model.named_parameters()
        if p.requires_grad and _is_lora_param(name)
    ]


def get_lora_param_dim(model: nn.Module) -> int:
    """Total number of trainable LoRA scalar parameters."""
    return sum(
        p.numel()
        for name, p in model.named_parameters()
        if p.requires_grad and _is_lora_param(name)
    )


def extract_lora_gradient(model: nn.Module) -> Tensor:
    """Concatenate the ``.grad`` of every trainable LoRA parameter into a flat vector.

    Must be called *after* ``loss.backward()`` and *before* ``optimizer.zero_grad()``.
    Returns a 1-D CPU float32 tensor.
    """
    grads: list[Tensor] = []
    for name, p in model.named_parameters():
        if p.requires_grad and _is_lora_param(name):
            if p.grad is None:
                grads.append(torch.zeros(p.numel(), dtype=torch.float32))
            else:
                grads.append(p.grad.detach().float().cpu().flatten())
    return torch.cat(grads)


def extract_lora_params(model: nn.Module) -> Tensor:
    """Concatenate current LoRA parameter *values* into a flat vector (detached, cpu, fp32)."""
    params: list[Tensor] = []
    for name, p in model.named_parameters():
        if p.requires_grad and _is_lora_param(name):
            params.append(p.detach().float().cpu().flatten())
    return torch.cat(params)


def set_lora_requires_grad(model: nn.Module, requires_grad: bool) -> None:
    """Toggle requires_grad on all LoRA parameters (useful for profiling)."""
    for name, p in model.named_parameters():
        if _is_lora_param(name):
            p.requires_grad_(requires_grad)
        else:
            p.requires_grad_(False)
