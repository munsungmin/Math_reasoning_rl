"""Conditional output distributions are data-dependent metrics, not model callbacks."""

from .base import Metric
from .functional.probability import categorical_entropy, categorical_kl
from .prediction_data import PredictionData


class ConditionalEntropy(Metric):
    required = frozenset({"sample_id", "log_probs"})

    def __init__(self):
        self.reset()

    def reset(self):
        self.data = PredictionData()

    def update(self, records):
        self.data.add(records)

    def compute(self):
        import torch

        rows = list(self.data.records.values())
        value = (
            categorical_entropy(
                torch.tensor([r["log_probs"] for r in rows], dtype=torch.float64)
            ).mean()
            if rows
            else None
        )
        return {"conditional_entropy": float(value) if value is not None else None}


class ConditionalKL(ConditionalEntropy):
    required = frozenset({"sample_id", "log_probs", "reference_log_probs"})

    def compute(self):
        import torch

        rows = list(self.data.records.values())
        value = (
            categorical_kl(
                torch.tensor([r["log_probs"] for r in rows], dtype=torch.float64),
                torch.tensor(
                    [r["reference_log_probs"] for r in rows], dtype=torch.float64
                ),
            ).mean()
            if rows
            else None
        )
        return {"conditional_kl": float(value) if value is not None else None}
