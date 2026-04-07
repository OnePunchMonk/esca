"""Core data structures for capability profiles."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from safetensors.numpy import save_file, load_file


@dataclass
class CapabilitySubspace:
    """The geometric representation of a single capability in LoRA parameter space."""

    name: str
    basis_vectors: np.ndarray  # (subspace_rank, lora_param_dim)
    singular_values: np.ndarray  # (subspace_rank,)
    variance_explained: float
    effective_rank: int
    baseline_loss: float
    eval_set_size: int
    layer_contributions: Optional[Dict[str, float]] = None


@dataclass
class CapabilityProfile:
    """Complete capability fingerprint of a model.

    Stores the subspace geometry for every profiled capability,
    the pairwise overlap matrix, and metadata needed for downstream
    prediction / protection / auditing.
    """

    subspaces: Dict[str, CapabilitySubspace]
    model_name: str
    lora_config: Dict
    lora_param_dim: int
    total_model_params: int

    # Pre-computed overlap matrix (filled by profiler)
    overlap_matrix: Optional[np.ndarray] = None
    capability_names: list[str] = field(default_factory=list)

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    sentinel_version: str = "0.0.1"
    compute_time_seconds: float = 0.0
    device_info: str = ""

    # ---- persistence ---------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialize profile to disk.

        Layout:
            <path>/
                metadata.json   — scalar / dict fields
                tensors.safetensors — all numpy arrays keyed by capability
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # -- tensors --
        tensor_dict: dict[str, np.ndarray] = {}
        for name, sub in self.subspaces.items():
            tensor_dict[f"{name}__basis"] = sub.basis_vectors.astype(np.float32)
            tensor_dict[f"{name}__singular_values"] = sub.singular_values.astype(
                np.float32
            )
        if self.overlap_matrix is not None:
            tensor_dict["__overlap_matrix__"] = self.overlap_matrix.astype(np.float32)
        save_file(tensor_dict, str(path / "tensors.safetensors"))

        # -- metadata --
        meta: dict = {
            "model_name": self.model_name,
            "lora_config": self.lora_config,
            "lora_param_dim": self.lora_param_dim,
            "total_model_params": self.total_model_params,
            "capability_names": self.capability_names,
            "created_at": self.created_at,
            "sentinel_version": self.sentinel_version,
            "compute_time_seconds": self.compute_time_seconds,
            "device_info": self.device_info,
            "subspace_meta": {},
        }
        for name, sub in self.subspaces.items():
            meta["subspace_meta"][name] = {
                "variance_explained": sub.variance_explained,
                "effective_rank": sub.effective_rank,
                "baseline_loss": sub.baseline_loss,
                "eval_set_size": sub.eval_set_size,
            }
        with open(path / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "CapabilityProfile":
        """Deserialize a profile from disk."""
        path = Path(path)
        with open(path / "metadata.json") as f:
            meta = json.load(f)

        tensors = load_file(str(path / "tensors.safetensors"))

        subspaces: Dict[str, CapabilitySubspace] = {}
        for name in meta["capability_names"]:
            sm = meta["subspace_meta"][name]
            subspaces[name] = CapabilitySubspace(
                name=name,
                basis_vectors=tensors[f"{name}__basis"],
                singular_values=tensors[f"{name}__singular_values"],
                variance_explained=sm["variance_explained"],
                effective_rank=sm["effective_rank"],
                baseline_loss=sm["baseline_loss"],
                eval_set_size=sm["eval_set_size"],
            )

        overlap = tensors.get("__overlap_matrix__")

        return cls(
            subspaces=subspaces,
            model_name=meta["model_name"],
            lora_config=meta["lora_config"],
            lora_param_dim=meta["lora_param_dim"],
            total_model_params=meta["total_model_params"],
            overlap_matrix=overlap,
            capability_names=meta["capability_names"],
            created_at=meta["created_at"],
            sentinel_version=meta["sentinel_version"],
            compute_time_seconds=meta["compute_time_seconds"],
            device_info=meta["device_info"],
        )

    # ---- display -------------------------------------------------------------

    def summary(self) -> str:
        lines = [
            f"CapabilityProfile — {self.model_name}",
            f"  LoRA dim: {self.lora_param_dim:,}  |  Model params: {self.total_model_params:,}",
            f"  Capabilities ({len(self.subspaces)}):",
        ]
        for name, sub in self.subspaces.items():
            lines.append(
                f"    {name:20s}  eff_rank={sub.effective_rank:<4d}  "
                f"var_explained={sub.variance_explained:.3f}  "
                f"baseline_loss={sub.baseline_loss:.4f}  "
                f"n_eval={sub.eval_set_size}"
            )
        return "\n".join(lines)

    def overlap_report(self) -> str:
        if self.overlap_matrix is None:
            return "No overlap matrix computed."
        names = self.capability_names
        header = "            " + "".join(f"{n:>12s}" for n in names)
        lines = [header]
        for i, name in enumerate(names):
            row = f"{name:12s}" + "".join(
                f"{self.overlap_matrix[i, j]:12.3f}" for j in range(len(names))
            )
            lines.append(row)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()
