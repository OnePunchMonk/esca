"""CapabilityProfiler — compute capability subspaces for a LoRA-adapted model."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Optional, Sequence, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .profile import CapabilityProfile, CapabilitySubspace
from ..utils.lora_utils import (
    extract_lora_gradient,
    get_lora_param_dim,
    set_lora_requires_grad,
)
from ..utils.svd_utils import randomized_svd, subspace_overlap

logger = logging.getLogger("sentinel.profiler")


class CapabilityProfiler:
    """Compute capability subspaces for each eval set via SVD of per-example LoRA gradients.

    Usage::

        profiler = CapabilityProfiler(model, tokenizer)
        profile = profiler.profile(capabilities={"math": math_dataset, ...})
        profile.save("my-profile.sentinel")
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        subspace_rank: int = 64,
        gradient_batch_size: int = 4,
        max_examples_per_capability: int = 500,
        max_seq_length: int = 512,
        seed: int = 42,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.subspace_rank = subspace_rank
        self.gradient_batch_size = gradient_batch_size
        self.max_examples = max_examples_per_capability
        self.max_seq_length = max_seq_length
        self.seed = seed
        self.device = device
        self.dtype = dtype

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------

    def profile(
        self,
        capabilities: Dict[str, Any],
    ) -> CapabilityProfile:
        """Profile every capability and return a ``CapabilityProfile``.

        Parameters
        ----------
        capabilities : mapping of ``capability_name → dataset``.
            Each dataset should be iterable and yield dicts with at least
            a ``"text"`` or ``"input_ids"`` key (standard HF dataset rows).
        """
        t0 = time.time()
        lora_dim = get_lora_param_dim(self.model)
        total_params = sum(p.numel() for p in self.model.parameters())

        lora_config = self._extract_lora_config()

        logger.info(
            "Profiling %s — LoRA dim=%d, model params=%d",
            getattr(self.model, "name_or_path", self.model.__class__.__name__),
            lora_dim,
            total_params,
        )

        set_lora_requires_grad(self.model, True)
        self.model.eval()

        subspaces: Dict[str, CapabilitySubspace] = {}
        cap_names: list[str] = []

        for cap_name, dataset in capabilities.items():
            logger.info("Computing subspace for '%s' ...", cap_name)
            t_cap = time.time()
            sub = self._compute_subspace(cap_name, dataset, lora_dim)
            elapsed = time.time() - t_cap
            logger.info(
                "  '%s' done [%.1fs] — eff_rank=%d, var_explained=%.3f",
                cap_name,
                elapsed,
                sub.effective_rank,
                sub.variance_explained,
            )
            subspaces[cap_name] = sub
            cap_names.append(cap_name)

        # Overlap matrix
        overlap = self._compute_overlap_matrix(subspaces, cap_names)

        device_info = str(self.device)
        if torch.cuda.is_available() and "cuda" in str(self.device):
            device_info = torch.cuda.get_device_name(0)

        profile = CapabilityProfile(
            subspaces=subspaces,
            model_name=getattr(
                self.model.config, "_name_or_path", self.model.__class__.__name__
            ),
            lora_config=lora_config,
            lora_param_dim=lora_dim,
            total_model_params=total_params,
            overlap_matrix=overlap,
            capability_names=cap_names,
            compute_time_seconds=time.time() - t0,
            device_info=device_info,
        )

        logger.info(
            "Profile complete — %d capabilities, %.1fs total",
            len(subspaces),
            profile.compute_time_seconds,
        )
        return profile

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _extract_lora_config(self) -> Dict:
        """Best-effort extraction of LoRA config from a PEFT model."""
        try:
            peft_config = self.model.peft_config
            # peft_config is a dict of adapter_name → LoraConfig
            cfg = list(peft_config.values())[0]
            return {
                "r": cfg.r,
                "lora_alpha": cfg.lora_alpha,
                "target_modules": list(cfg.target_modules)
                if cfg.target_modules
                else [],
                "lora_dropout": cfg.lora_dropout,
            }
        except Exception:
            return {}

    @torch.no_grad()
    def _compute_baseline_loss(
        self, dataset: Any, max_examples: int = 100
    ) -> float:
        """Average cross-entropy loss on a small sample of the eval set."""
        self.model.eval()
        total_loss = 0.0
        count = 0
        for i, example in enumerate(dataset):
            if i >= max_examples:
                break
            inputs = self._tokenize(example)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            total_loss += outputs.loss.item()
            count += 1
        return total_loss / max(count, 1)

    def _compute_subspace(
        self,
        cap_name: str,
        dataset: Any,
        lora_dim: int,
    ) -> CapabilitySubspace:
        """Collect per-example LoRA gradients, run SVD, return subspace."""
        # Collect gradients
        gradients: list[Tensor] = []
        total_loss = 0.0
        count = 0

        for i, example in enumerate(dataset):
            if i >= self.max_examples:
                break

            self.model.zero_grad()
            inputs = self._tokenize(example)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Enable grad for this forward-backward
            with torch.enable_grad():
                outputs = self.model(**inputs)
                loss = outputs.loss
                loss.backward()

            grad = extract_lora_gradient(self.model)
            gradients.append(grad)
            total_loss += loss.item()
            count += 1

        if count == 0:
            raise ValueError(f"No examples in dataset for capability '{cap_name}'")

        # Stack into gradient matrix: (n_examples, lora_dim)
        G = torch.stack(gradients)  # (n, d)

        # Determine rank for SVD
        rank = min(self.subspace_rank, G.shape[0] - 1, G.shape[1])
        if rank < 1:
            rank = 1

        # Randomized SVD
        # We want the right singular vectors (directions in parameter space)
        # G = U S Vt  →  Vt rows are the principal gradient directions
        U, S, Vt = randomized_svd(G, rank=rank, seed=self.seed)

        # Variance explained
        total_var = float((G.numpy() ** 2).sum())
        captured_var = float((S ** 2).sum())
        variance_explained = captured_var / (total_var + 1e-12)

        # Effective rank: directions with >1% of max singular value
        effective_rank = int((S > S[0] * 0.01).sum())

        baseline_loss = total_loss / count

        return CapabilitySubspace(
            name=cap_name,
            basis_vectors=Vt,  # (rank, lora_dim)
            singular_values=S,
            variance_explained=variance_explained,
            effective_rank=effective_rank,
            baseline_loss=baseline_loss,
            eval_set_size=count,
        )

    def _compute_overlap_matrix(
        self,
        subspaces: Dict[str, CapabilitySubspace],
        names: list[str],
    ) -> np.ndarray:
        """Pairwise subspace overlap matrix."""
        n = len(names)
        overlap = np.eye(n, dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                ov = subspace_overlap(
                    subspaces[names[i]].basis_vectors,
                    subspaces[names[j]].basis_vectors,
                )
                overlap[i, j] = ov
                overlap[j, i] = ov
        return overlap

    def _tokenize(self, example: dict) -> dict:
        """Tokenize a single example. Expects ``text`` or ``input_ids`` key."""
        if "input_ids" in example:
            ids = example["input_ids"]
            if not isinstance(ids, Tensor):
                ids = torch.tensor(ids, dtype=torch.long)
            ids = ids[: self.max_seq_length].unsqueeze(0)
            return {"input_ids": ids, "labels": ids.clone()}

        text = example.get("text") or example.get("content") or ""
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length,
        )
        enc["labels"] = enc["input_ids"].clone()
        return enc
