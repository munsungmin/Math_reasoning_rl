from collections import defaultdict

from .base import Metric
from .functional.probability import pass_at_k
from .prediction_data import PredictionData


class PassAtK(Metric):
    required = frozenset({"sample_id", "sample_index", "correct"})

    def __init__(self, ks=(1,)):
        self.ks = tuple(ks)
        if not self.ks or any(type(k) is not int or k < 1 for k in self.ks):
            raise ValueError("ks must contain positive integers")
        self.reset()

    def reset(self):
        self.data = PredictionData()

    def update(self, records):
        self.data.add(records)

    def compute(self):
        groups = defaultdict(list)
        for key, row in self.data.records.items():
            groups[key[:3]].append(bool(row["correct"]))
        return {
            f"pass@{k}": sum(pass_at_k(len(v), sum(v), k) for v in groups.values())
            / len(groups)
            if groups
            else None
            for k in self.ks
        }
