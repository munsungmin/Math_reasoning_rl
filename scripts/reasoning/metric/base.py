"""Data-dependent observations. No model loading, generation or training here."""


class Metric:
    required = frozenset()

    def update(self, records):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class MetricCollection:
    def __init__(self, metrics):
        self.metrics = dict(metrics)

    def update(self, records):
        records = list(records)
        for name, metric in self.metrics.items():
            for row in records:
                missing = metric.required - row.keys()
                if missing:
                    raise ValueError(f"Metric {name} requires {sorted(missing)}")
        for metric in self.metrics.values():
            metric.update(records)

    def compute(self):
        result = {}
        for name, metric in self.metrics.items():
            values = metric.compute()
            for key, value in values.items():
                if key in result:
                    raise ValueError(f"Duplicate metric output {key} from {name}")
                result[key] = value
        return result

    def reset(self):
        for metric in self.metrics.values():
            metric.reset()
