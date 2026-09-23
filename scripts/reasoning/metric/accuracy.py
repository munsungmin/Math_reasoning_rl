from collections import Counter

from .base import Metric
from .prediction_data import PredictionData


class Accuracy(Metric):
    required = frozenset({"sample_id", "correct"})

    def __init__(self):
        self.reset()

    def reset(self):
        self.data = PredictionData()

    def update(self, records):
        self.data.add(records)

    def compute(self):
        rows = list(self.data.records.values())
        n = len(rows)
        c = sum(bool(row["correct"]) for row in rows)
        return {
            "accuracy": c / n if n else None,
            "correct": c,
            "samples": n,
            "extraction_status_counts": dict(
                Counter(r.get("extraction_status", "unknown") for r in rows)
            ),
        }


class LengthStatistics(Metric):
    required = frozenset({"sample_id", "generated_tokens", "hit_length_limit"})

    def __init__(self):
        self.reset()

    def reset(self):
        self.data = PredictionData()

    def update(self, records):
        self.data.add(records)

    def compute(self):
        rows = list(self.data.records.values())
        n = len(rows)
        return {
            "mean_generated_tokens": sum(r["generated_tokens"] for r in rows) / n
            if n
            else None,
            "length_limit_rate": sum(r["hit_length_limit"] for r in rows) / n
            if n
            else None,
        }
