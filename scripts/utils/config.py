from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from .paths import REPO


def register_resolvers():
    from scripts.train.rl.native_config import register_resolvers as native_resolvers

    native_resolvers()
    OmegaConf.register_new_resolver(
        "seed_offset", lambda seed, offset: int(seed) + int(offset), replace=True
    )
    OmegaConf.register_new_resolver("list_len", len, replace=True)


def compose_experiment(overrides=()):
    register_resolvers()
    if GlobalHydra.instance().is_initialized():
        return compose(config_name="config", overrides=list(overrides))
    with initialize_config_dir(config_dir=str(REPO / "config"), version_base="1.3"):
        return compose(config_name="config", overrides=list(overrides))


def native_defaults():
    register_resolvers()
    if GlobalHydra.instance().is_initialized():
        return compose(config_name="ppo_trainer")
    with initialize_config_dir(
        config_dir=str(REPO / "ext/verl/verl/trainer/config"), version_base="1.3"
    ):
        return compose(config_name="ppo_trainer")


def plain(config):
    return OmegaConf.to_container(config, resolve=True)


def validate(config):
    if config.stage not in ("fit", "test", "predict"):
        raise ValueError(f"Unknown stage: {config.stage}")
    if config.command not in ("run", "check"):
        raise ValueError(f"Unknown command: {config.command}")
    if len(set(config.runtime.gpu_ids)) != len(config.runtime.gpu_ids):
        raise ValueError("runtime.gpu_ids must be unique")
    if config.stage == "fit" and config.task.kind not in ("rl", "solve", "classify"):
        raise ValueError("Unknown training task")
    if config.task.kind == "classify":
        if len(config.task.labels) < 2:
            raise ValueError("Classification requires at least two labels")
        if config.data.classification == "multi" and len(config.task.labels) != 4:
            raise ValueError("PERL multi classification requires four label tokens")
        if (
            config.data.classification in ("binary", "single")
            and len(config.task.labels) != 2
        ):
            raise ValueError("Binary/single classification requires two label tokens")
        if list(config.task.labels) != list("ABCD"[: len(config.task.labels)]):
            raise ValueError(
                "The PERL prompt/target view uses label codes A/B/C/D (multi) or A/B (binary/single)"
            )
    # Compose/resolve before starting workers; configured metric/callback targets are checked too.
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    from .callbacks import build_observations
    from .metric import build_metrics

    for key in ("val_metrics", "test_metrics"):
        build_metrics(config.task.get(key, {}))
    build_observations(config.callbacks)
