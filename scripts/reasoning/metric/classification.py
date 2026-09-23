from .base import Metric
from .prediction_data import PredictionData


class ClassificationMetrics(Metric):
    required = frozenset({"sample_id", "prediction", "label"})

    def __init__(self, num_classes):
        self.num_classes = int(num_classes)
        if self.num_classes < 2:
            raise ValueError("Classification requires at least two classes")
        self.reset()

    def reset(self):
        self.data = PredictionData()

    def update(self, records):
        rows = list(records)
        for row in rows:
            if any(
                not 0 <= int(row[k]) < self.num_classes for k in ("prediction", "label")
            ):
                raise ValueError(
                    "Classification label is outside the configured label space"
                )
        self.data.add(rows)

    def compute(self):
        matrix = [[0] * self.num_classes for _ in range(self.num_classes)]
        for row in self.data.records.values():
            matrix[int(row["label"])][int(row["prediction"])] += 1
        n = len(self.data.records)
        f1 = []
        for i in range(self.num_classes):
            denominator = sum(matrix[i]) + sum(row[i] for row in matrix)
            f1.append(2 * matrix[i][i] / denominator if denominator else 0.0)
        return {
            "accuracy": sum(matrix[i][i] for i in range(self.num_classes)) / n
            if n
            else None,
            "macro_f1": sum(f1) / len(f1) if n else None,
            "confusion_matrix": matrix,
            "samples": n,
        }
