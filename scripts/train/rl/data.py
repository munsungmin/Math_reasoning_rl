"""Extracted native verl data; legacy run semantics are preserved."""

import hashlib
import uuid
from pathlib import Path

from scripts.reasoning.data.schema import check_disjoint
from scripts.utils.io import read_rows, write_json

from .native_config import build_config


def prepare_run(run, seed=None, settings=None):
    """Freeze disjoint train/validation/test copies before starting any GPU worker."""
    import pandas as pd
    from omegaconf import OmegaConf

    cfg = build_config(run, seed=seed, settings=settings)
    sources = {name: Path(path) for name, path in cfg.launcher.dataset.sources.items()}
    if set(sources) != {"train", "validation", "test"}:
        raise ValueError("train/validation/test 원본 경로가 모두 필요합니다.")
    splits = {name: read_rows(path) for name, path in sources.items()}
    check_disjoint(*splits.values())
    counts = {name: len(rows) for name, rows in splits.items()}
    if (
        cfg.launcher.dataset.expected_counts is not None
        and counts != dict(cfg.launcher.dataset.expected_counts)
    ) or min(counts.values()) <= 0:
        raise ValueError(f"동결 데이터 개수가 설정과 다릅니다: {counts}")
    train_count = counts["train"]
    if cfg.data.train_max_samples > 0:
        train_count = min(train_count, cfg.data.train_max_samples)
    available_steps = (
        train_count // cfg.data.train_batch_size
    ) * cfg.trainer.total_epochs
    if available_steps < cfg.trainer.total_training_steps:
        raise ValueError("total_epochs가 목표 updates를 채우기에 부족합니다.")
    for name in ("train", "validation"):
        key = "train_files" if name == "train" else "val_files"
        if list(cfg.data[key]) != [str(run / f"{name}.parquet")]:
            raise ValueError("데이터 원본은 launcher.dataset.sources에서 변경하세요.")
    run.mkdir(parents=True, exist_ok=False)
    checkpoints = Path(cfg.trainer.default_local_dir)
    initialization = None
    if "math_reasoning" in cfg and cfg.actor_rollout_ref.model.lora_adapter_path:
        from .adapter import prepare_initial_adapter

        initialization = prepare_initial_adapter(
            cfg.actor_rollout_ref.model.lora_adapter_path,
            run / "initial_adapter",
            cfg.actor_rollout_ref.model.path,
        )
        cfg.actor_rollout_ref.model.lora_adapter_path = initialization["path"]
    for name, rows in splits.items():
        records = []
        for index, row in enumerate(rows):
            if (
                cfg.launcher.dataset.levels is not None
                and row["level"] not in cfg.launcher.dataset.levels
            ):
                raise ValueError("데이터 level이 launcher.dataset.levels에 없습니다.")
            records.append(
                {
                    "data_source": f"{cfg.launcher.dataset.name}/{name}",
                    "prompt": [{"role": "user", "content": row["prompt"]}],
                    "reward_model": {"style": "rule", "ground_truth": row["answer"]},
                    "extra_info": {
                        "sample_id": row["sample_id"],
                        "index": index,
                        "level": row["level"],
                    },
                }
            )
        pd.DataFrame(records).to_parquet(run / f"{name}.parquet", index=False)
    OmegaConf.save(cfg, run / "config.yaml")
    manifest = {
        "framework": "ext/verl",
        "seed": cfg.launcher.seed,
        "checkpoint_base": cfg.launcher.checkpoints_dir,
        "checkpoints": str(checkpoints),
        "model": cfg.actor_rollout_ref.model.path,
        "gpu_ids": ",".join(map(str, cfg.launcher.gpu_ids)),
        "counts": counts,
        "source_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in sources.items()
        },
        "parquet_sha256": {
            name: hashlib.sha256((run / f"{name}.parquet").read_bytes()).hexdigest()
            for name in sources
        },
        "wandb_run_id": uuid.uuid4().hex[:12],
        "wandb_entity": cfg.launcher.wandb.entity,
        "settings_basis": "Effective configuration frozen in config.yaml before training.",
    }
    if "math_reasoning" in cfg:
        from scripts.reasoning.model import ModelSpec
        from scripts.utils.manifest import model_manifest, source_manifest

        project = cfg.math_reasoning
        OmegaConf.save(project, run / "experiment.yaml")
        hydra_dir = run / ".hydra"
        hydra_dir.mkdir()
        OmegaConf.save(project, hydra_dir / "config.yaml")
        manifest["provenance"] = source_manifest(run / "source")
        manifest["initialization"] = model_manifest(
            ModelSpec.resolve(project.model.initialization)
        )
        manifest["adapter_initialization"] = initialization
    write_json(run / "manifest.json", manifest)
    return cfg, manifest
