"""SurgeryPlan — a concrete plan to clean a training dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SurgeryPlan:
    """Concrete, executable remediation plan for a training dataset.

    Call :meth:`apply` to get a modified dataset, or use the
    individual methods for surgical operations.
    """

    # Which examples to remove (by index in original dataset)
    examples_to_remove: List[int] = field(default_factory=list)

    # Per-example loss weights (example_id → weight, default=1.0)
    example_weights: Dict[int, float] = field(default_factory=dict)

    # Augmentation: how many retention examples per capability
    retention_requests: Dict[str, int] = field(default_factory=dict)

    # Human-readable summary of what the plan does
    summary_text: str = ""

    # Predicted outcomes
    predicted_regression_recovery: Dict[str, float] = field(default_factory=dict)
    predicted_target_task_cost: float = 0.0

    def summary(self) -> str:
        lines = [
            "╔══════════════════════════════════════════╗",
            "║         SENTINEL SURGERY PLAN            ║",
            "╠══════════════════════════════════════════╣",
        ]
        if self.examples_to_remove:
            lines.append(f"║  Remove: {len(self.examples_to_remove)} examples" + " " * (31 - len(str(len(self.examples_to_remove)))) + "║")
        if self.example_weights:
            n_reweighted = sum(1 for w in self.example_weights.values() if w != 1.0)
            lines.append(f"║  Reweight: {n_reweighted} examples" + " " * (29 - len(str(n_reweighted))) + "║")
        for cap, n in self.retention_requests.items():
            lines.append(f"║  Augment '{cap}': +{n} retention examples" + " " * max(0, 26 - len(cap) - len(str(n))) + "║")
        if self.predicted_regression_recovery:
            lines.append("║  Expected recovery:                      ║")
            for cap, rec in self.predicted_regression_recovery.items():
                lines.append(f"║    {cap:<16}: {rec:.0%}" + " " * 17 + "║")
        lines.append(f"║  Target task cost: {self.predicted_target_task_cost:.2%}" + " " * 20 + "║")
        lines.append("╚══════════════════════════════════════════╝")
        if self.summary_text:
            lines.append(f"→ {self.summary_text}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------

    def remove_harmful(self, dataset: Any) -> Any:
        """Return a new dataset with harmful examples removed.

        Works with HuggingFace Datasets or any list-like object.
        """
        remove_set = set(self.examples_to_remove)
        if hasattr(dataset, "filter"):
            # HuggingFace Dataset
            def keep(example: dict, idx: int) -> bool:
                return idx not in remove_set

            return dataset.filter(keep, with_indices=True)
        else:
            return [ex for i, ex in enumerate(dataset) if i not in remove_set]

    def reweight(self, dataset: Any) -> tuple[Any, Dict[int, float]]:
        """Return the original dataset plus a per-example weight dict.

        The caller is responsible for applying these weights to the loss
        (e.g., via a custom data collator or loss weighting hook).
        """
        return dataset, dict(self.example_weights)

    def augment_with_retention(
        self, dataset: Any, retention_datasets: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Return dataset augmented with retention examples.

        Parameters
        ----------
        retention_datasets
            Mapping of capability → pre-loaded dataset.  If None, a
            warning is logged and the original dataset is returned unchanged.
        """
        import logging

        logger = logging.getLogger("sentinel.surgeon")

        if retention_datasets is None:
            logger.warning(
                "No retention datasets provided — cannot augment. "
                "Pass retention_datasets={cap: dataset} to augment_with_retention()."
            )
            return dataset

        additional: list = []
        for cap, n_req in self.retention_requests.items():
            ret_data = retention_datasets.get(cap)
            if ret_data is None:
                logger.warning("No retention dataset for '%s' — skipping.", cap)
                continue
            n_avail = len(ret_data) if hasattr(ret_data, "__len__") else 0
            n = min(n_req, n_avail) if n_avail > 0 else n_req
            sample = list(ret_data)[:n]
            additional.extend(sample)

        if not additional:
            return dataset

        if hasattr(dataset, "concatenate"):
            import datasets as ds_lib

            return ds_lib.concatenate_datasets(
                [dataset, ds_lib.Dataset.from_list(additional)]
            )
        else:
            return list(dataset) + additional

    def apply(
        self,
        dataset: Any,
        retention_datasets: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Apply all surgery operations: remove → reweight → augment."""
        result = self.remove_harmful(dataset)
        result = self.augment_with_retention(result, retention_datasets)
        return result

    def to_json(self, path: str) -> None:
        """Save plan to JSON."""
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "examples_to_remove": self.examples_to_remove,
                    "example_weights": {str(k): v for k, v in self.example_weights.items()},
                    "retention_requests": self.retention_requests,
                    "summary": self.summary_text,
                    "predicted_regression_recovery": self.predicted_regression_recovery,
                    "predicted_target_task_cost": self.predicted_target_task_cost,
                },
                fh,
                indent=2,
            )

    @classmethod
    def from_json(cls, path: str) -> "SurgeryPlan":
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(
            examples_to_remove=data.get("examples_to_remove", []),
            example_weights={int(k): v for k, v in data.get("example_weights", {}).items()},
            retention_requests=data.get("retention_requests", {}),
            summary_text=data.get("summary", ""),
            predicted_regression_recovery=data.get("predicted_regression_recovery", {}),
            predicted_target_task_cost=data.get("predicted_target_task_cost", 0.0),
        )
