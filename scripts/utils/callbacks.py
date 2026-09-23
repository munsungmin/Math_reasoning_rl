from hydra.utils import instantiate

from scripts.reasoning.callbacks.base import ModelObservation, Observations


def build_observations(config):
    callbacks = {name: instantiate(spec) for name, spec in (config or {}).items()}
    for name, callback in callbacks.items():
        if not isinstance(callback, ModelObservation):
            raise TypeError(f"{name} is not a model observation callback")
    return Observations(callbacks)
