from hydra.utils import instantiate

from scripts.reasoning.metric.base import MetricCollection


def build_metrics(config):
    return MetricCollection(
        {name: instantiate(spec) for name, spec in (config or {}).items()}
    )
