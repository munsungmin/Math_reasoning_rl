"""Freeze effective recipes, input identities and executable source provenance."""

import json
import platform
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import OmegaConf

from .io import sha256, write_json
from .paths import REPO


def source_manifest(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    files = [REPO / "main.py"]
    for folder in ("scripts", "config", "script", "configs"):
        files += [
            p
            for p in (REPO / folder).rglob("*")
            if p.is_file() and p.suffix in (".py", ".yaml")
        ]
    hashes = {}
    for path in files:
        relative = path.relative_to(REPO)
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(relative)] = sha256(path)
    revisions = {}
    for name, repo in (
        ("project", REPO),
        ("verl", REPO / "ext/verl"),
        ("grader", REPO / "ext/limit-of-RLVR"),
    ):
        revisions[name] = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        with (directory / f"{name}.patch").open("w") as stream:
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo),
                    "diff",
                    "HEAD",
                    "--",
                    "*.py",
                    "*.yaml",
                    "*.toml",
                ],
                stdout=stream,
                check=True,
                text=True,
            )
    return {
        "revisions": revisions,
        "source_sha256": hashes,
        "patch_sha256": {
            name: sha256(directory / f"{name}.patch") for name in revisions
        },
    }


def model_manifest(spec):
    locations = {
        "base": spec.path,
        "adapter": spec.adapter,
        "tokenizer": spec.tokenizer,
    }
    hashes = {}
    for directory in set(locations.values()) - {None}:
        for path in Path(directory).iterdir():
            if path.is_file() and path.suffix in (
                ".json",
                ".safetensors",
                ".bin",
                ".model",
                ".txt",
            ):
                hashes[str(path)] = sha256(path)
    return {"locations": locations, "sha256": hashes}


def validate_model_manifest(manifest):
    for path, expected in manifest["sha256"].items():
        if sha256(path) != expected:
            raise ValueError(f"Initial model/tokenizer changed: {path}")


def validate_provenance(provenance):
    for relative, expected in provenance["source_sha256"].items():
        if not (REPO / relative).exists() or sha256(REPO / relative) != expected:
            raise ValueError(
                f"Resume source changed: {relative}; restore the saved run/source version"
            )
    for name, folder in (("verl", "ext/verl"), ("grader", "ext/limit-of-RLVR")):
        current = subprocess.check_output(
            ["git", "-C", str(REPO / folder), "rev-parse", "HEAD"], text=True
        ).strip()
        if current != provenance["revisions"][name]:
            raise ValueError(f"Resume dependency revision changed: {name}")
        import hashlib

        patch = subprocess.check_output(
            [
                "git",
                "-C",
                str(REPO / folder),
                "diff",
                "HEAD",
                "--",
                "*.py",
                "*.yaml",
                "*.toml",
            ]
        )
        if hashlib.sha256(patch).hexdigest() != provenance["patch_sha256"][name]:
            raise ValueError(f"Resume dependency working tree changed: {name}")


def prepare_run(config, *, overrides=()):
    from hydra.utils import instantiate

    from scripts.reasoning.model import ModelSpec

    data = instantiate(config.data)
    if config.evaluation.predictions:
        prediction_path = Path(config.evaluation.predictions).resolve()
        data_manifest = {
            "sources": {
                "predictions": {
                    "path": str(prediction_path),
                    "sha256": sha256(prediction_path),
                }
            }
        }
    elif config.stage == "fit":
        data.validate_splits()
    else:
        data.rows(config.evaluation.split)
    if not config.evaluation.predictions:
        data_manifest = data.manifest()
    spec = (
        ModelSpec.resolve(config.model.initialization)
        if not config.evaluation.predictions
        else None
    )
    name = (
        datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S-") + uuid.uuid4().hex[:8]
    )
    output = Path(
        config.runtime.output_dir
        or Path(config.runtime.base_dir)
        / "artifacts/composable"
        / config.experiment_name
        / name
    )
    output.mkdir(parents=True, exist_ok=False)
    config.runtime.output_dir = str(output.resolve())
    frozen = OmegaConf.to_container(config, resolve=True)
    OmegaConf.save(OmegaConf.create(frozen), output / "config.yaml")
    hydra_dir = output / ".hydra"
    hydra_dir.mkdir()
    OmegaConf.save(OmegaConf.create(frozen), hydra_dir / "config.yaml")
    OmegaConf.save(OmegaConf.create(list(overrides)), hydra_dir / "overrides.yaml")
    manifest = {
        "schema_version": 1,
        "task": config.task.kind,
        "stage": config.stage,
        "model": model_manifest(spec) if spec else None,
        "data": data_manifest,
        "provenance": source_manifest(output / "source"),
    }
    write_json(output / "manifest.json", manifest)
    return output


def record_environment(output):
    from importlib.metadata import distributions

    write_json(
        Path(output) / "environment.json",
        {
            "python": sys.executable,
            "version": sys.version,
            "platform": platform.platform(),
            "packages": sorted(
                {
                    f"{d.metadata['Name']}=={d.version}"
                    for d in distributions()
                    if d.metadata.get("Name")
                }
            ),
        },
    )


def validate_resume(run):
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text())
    validate_provenance(manifest["provenance"])
    if manifest["model"]:
        validate_model_manifest(manifest["model"])
    for source in manifest["data"]["sources"].values():
        if sha256(source["path"]) != source["sha256"]:
            raise ValueError(f"Resume input changed: {source['path']}")
    return OmegaConf.load(run / "config.yaml")
