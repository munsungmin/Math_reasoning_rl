"""Hydra entry point for independently composed RL, SFT and evaluation jobs."""

import os
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf

from scripts.utils.config import register_resolvers, validate
from scripts.utils.paths import REPO


def dispatch(config):
    validate(config)
    if config.resume:
        if config.command == "check":
            raise ValueError(
                "Use command=run for resume; command=check validates a new recipe"
            )
        if config.stage != "fit":
            raise ValueError(
                "resume is a training operation; evaluation takes model.initialization"
            )
        from hydra.core.override_parser.overrides_parser import OverridesParser

        overrides = OverridesParser.create().parse_overrides(
            HydraConfig.get().overrides.task
        )
        changed = [
            o.key_or_group
            for o in overrides
            if o.key_or_group not in ("resume", "experiment", "command")
        ]
        if changed:
            raise ValueError(
                f"Resume uses frozen configuration; remove overrides: {changed}"
            )
        run = Path(config.resume).resolve()
        if config.task.kind == "rl":
            from scripts.train.rl.runner import run_command

            return run_command("resume", selected_run=run)
        from scripts.utils.manifest import validate_resume

        config = validate_resume(run)
        import json

        if (run / "result.json").exists() and json.loads(
            (run / "result.json").read_text()
        ).get("completed"):
            print(f"This SFT run is complete: {run}")
            return
        config.resume = str(run / "checkpoints/last.ckpt")
        if not Path(config.resume).is_file():
            raise FileNotFoundError(config.resume)
        config.runtime.output_dir = str(run)
    elif config.stage == "fit" and config.task.kind == "rl":
        return instantiate(config.task).run(config)
    elif config.command == "check":
        from scripts.reasoning.model import ModelSpec

        data = instantiate(config.data)
        if config.evaluation.predictions:
            from scripts.utils.io import read_rows

            read_rows(config.evaluation.predictions)
        else:
            data.validate_splits() if config.stage == "fit" else data.rows(
                config.evaluation.split
            )
            ModelSpec.resolve(config.model.initialization)
        print(
            "Configuration, source splits and model artifact checked; no GPU job started."
        )
        return
    else:
        from scripts.utils.manifest import prepare_run

        prepare_run(config, overrides=HydraConfig.get().overrides.task)
    worker = (
        "scripts.train.sft.worker" if config.stage == "fit" else "scripts.eval.worker"
    )
    output = Path(config.runtime.output_dir)
    session = output / "sessions"
    session.mkdir(exist_ok=True)
    import uuid

    config_path = session / f"{uuid.uuid4().hex[:12]}.yaml"
    OmegaConf.save(config, config_path, resolve=True)
    environment = {
        **os.environ,
        **dict(config.runtime.worker_env),
        "CUDA_VISIBLE_DEVICES": ",".join(map(str, config.runtime.gpu_ids)),
    }
    if config.stage != "fit" and config.evaluation.predictions:
        environment["CUDA_VISIBLE_DEVICES"] = ""
    elif config.runtime.gpu_ids:
        from scripts.train.rl.environment import gpu_preflight

        gpu_preflight(environment["CUDA_VISIBLE_DEVICES"])
    from scripts.utils.process import run_worker_command

    run_worker_command(
        [config.runtime.python, "-m", worker, str(config_path)],
        cwd=REPO,
        env=environment,
    )


@hydra.main(version_base="1.3", config_path=str(REPO / "config"), config_name="config")
def hydra_main(config):
    dispatch(config)


def main():
    register_resolvers()
    hydra_main()
