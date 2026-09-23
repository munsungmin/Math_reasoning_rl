"""Extracted native verl runner; legacy run semantics are preserved."""

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from scripts.utils.paths import REPO

DEFAULT_CONFIG = REPO / "configs/verl_math.yaml"
BASE = Path("/data/sungmin/math_reasoning")
RUNS = BASE / "artifacts/math_error_transfer/verl/E1"
CHECKPOINTS = BASE / "checkpoints/math_error_transfer/verl/E1"
GRADER_PY = BASE / "envs/limit-rlvr-grader/bin/python"
WORLD_SIZE = 4
import hydra

from scripts.reasoning.data.schema import check_disjoint as check_disjoint
from scripts.reasoning.reward.verl import compute_score as compute_score
from scripts.utils.io import read_rows as read_rows
from scripts.utils.io import write_json

from .checkpoint import publish_last, validate_checkpoint
from .data import prepare_run
from .environment import check_environment, gpu_preflight
from .native_config import build_config as build_config
from .native_config import load_settings, register_resolvers, validate_launcher_config
from .worker import run_worker as run_worker


def read_manifest(run):
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest.get("framework") != "ext/verl":
        raise ValueError(
            "Native verl 실행만 재개할 수 있습니다. TRL checkpoint는 사용할 수 없습니다."
        )
    base = Path(manifest.get("checkpoint_base", CHECKPOINTS))
    expected = base / f"seed-{manifest['seed']:04d}" / run.name
    if Path(manifest["checkpoints"]).resolve() != expected.resolve():
        raise ValueError("Checkpoint 경로가 이 verl 실행 전용 경로와 다릅니다.")
    return manifest


def latest_run(seed, runs=RUNS):
    pointer = runs / f"seed-{seed:04d}" / "latest.json"
    return Path(json.loads(pointer.read_text())["run"])


def print_metric(event, total_steps):
    step, metrics = event["step"], event["data"]
    if "actor/loss" in metrics:
        print(
            f"{step:3}/{total_steps} | loss {metrics['actor/loss']:.4f} | "
            f"정답 {metrics.get('train/accuracy', float('nan')):.1%} | "
            f"reward {metrics['critic/rewards/mean']:.3f} | "
            f"update {metrics['timing_s/step']:.1f}s",
            flush=True,
        )
    for key, value in metrics.items():
        if key.startswith("val-core/") and "/acc/" in key:
            print(f"[평가] update {step} | {key} | 정답 {value:.1%}", flush=True)


def execute(run, cfg, manifest, phase):
    """Accumulate logs, publish last, and keep native training in its own child."""
    from omegaconf import OmegaConf

    checkpoints = Path(manifest["checkpoints"])
    options = cfg.get("launcher", {})
    gpus = manifest["gpu_ids"]
    world_size = cfg.trainer.n_gpus_per_node
    total_steps = cfg.trainer.total_training_steps
    keep_last_only = options.get("keep_last_only", True)
    session = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        + "-"
        + uuid.uuid4().hex[:6]
    )
    sessions = run / "sessions"
    sessions.mkdir(exist_ok=True)
    metrics_path = sessions / f"{session}.metrics.jsonl"
    config_path = sessions / f"{session}.yaml"
    cfg.trainer.rollout_data_dir = str(sessions / session / "rollouts")
    cfg.trainer.validation_data_dir = str(
        sessions / session / ("test" if phase == "test" else "validation")
    )
    OmegaConf.save(cfg, config_path)
    env = os.environ.copy()
    env.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="1",
        NCCL_P2P_DISABLE="1",
        NCCL_IB_DISABLE="1",
        VLLM_USE_FLASHINFER_SAMPLER="0",
        RAY_DEDUP_LOGS="0",
    )
    env.update(
        {key: str(value) for key, value in options.get("worker_env", {}).items()}
    )
    env.update(
        CUDA_VISIBLE_DEVICES=gpus,
        VERL_FILE_LOGGER_PATH=str(metrics_path),
        WANDB_MODE=os.environ.get(
            "WANDB_MODE", options.get("wandb", {}).get("mode", "online")
        ),
        WANDB_ENTITY=manifest["wandb_entity"],
        WANDB_RUN_ID=manifest["wandb_run_id"] + ("-test" if phase == "test" else ""),
        WANDB_RESUME="allow",
        WANDB_DIR=str(run),
    )
    print(
        f"[verl] {phase} | seed {manifest['seed']} | {total_steps} updates | GPU {gpus}\n"
        f"[경로] {run}\n[로그] {run / 'train.log'}",
        flush=True,
    )
    checkpoints.mkdir(parents=True, exist_ok=True)
    metrics_path.touch()
    with (
        (run / "train.log").open("ab", buffering=0) as log,
        metrics_path.open() as metrics,
        (run / "metrics.jsonl").open("a", buffering=1) as cumulative,
        (run / "rollouts.jsonl").open("a", buffering=1) as trajectory_log,
    ):
        log.write(f"\nSESSION {session} phase={phase}\n".encode())
        process = subprocess.Popen(
            [
                cfg.math_reasoning.runtime.python
                if "math_reasoning" in cfg
                else sys.executable,
                "-m",
                "scripts.train.rl.worker",
                str(config_path),
            ],
            cwd=REPO,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        last_status = time.monotonic()
        try:
            while True:
                finished = process.poll() is not None
                while True:
                    position = metrics.tell()
                    line = metrics.readline()
                    if not line.endswith("\n"):
                        metrics.seek(position)
                        break
                    event = json.loads(line)
                    event.update(session=session, phase=phase)
                    rollout_file = (
                        Path(cfg.trainer.rollout_data_dir) / f"{event['step']}.jsonl"
                    )
                    if phase != "test" and rollout_file.is_file():
                        rollouts = [
                            json.loads(row)
                            for row in rollout_file.read_text().splitlines()
                        ]
                        for row in rollouts:
                            trajectory_log.write(
                                json.dumps({**row, "session": session}) + "\n"
                            )
                        if rollouts:
                            event["data"]["train/accuracy"] = sum(
                                row["acc"] for row in rollouts
                            ) / len(rollouts)
                    cumulative.write(json.dumps(event) + "\n")
                    print_metric(event, total_steps)
                    if "math_reasoning" in cfg:
                        from scripts.utils.logger import ResultWriter

                        from .evaluation import evaluate_dump

                        validation = (
                            Path(cfg.trainer.validation_data_dir)
                            / f"{event['step']}.jsonl"
                        )
                        if validation.is_file():
                            split = "test" if phase == "test" else "validation"
                            metrics_config = (
                                cfg.math_reasoning.task.test_metrics
                                if phase == "test"
                                else cfg.math_reasoning.task.val_metrics
                            )
                            values = evaluate_dump(
                                validation,
                                metrics_config,
                                run / "evaluations" / session / f"step_{event['step']}",
                                split,
                            )
                            ResultWriter(run / "evaluation_metrics").log(
                                values, step=event["step"], stage=split
                            )
                published = publish_last(checkpoints, world_size, keep_last_only)
                if published is not None:
                    if "math_reasoning" in cfg and phase != "test":
                        from .observation import observe_checkpoint

                        observe_checkpoint(
                            run, cfg, published, final=published == total_steps
                        )
                    retention = (
                        "이전 저장본 삭제" if keep_last_only else "이전 저장본 보존"
                    )
                    print(
                        f"[저장] update {published} → last · {retention}",
                        flush=True,
                    )
                if finished:
                    break
                if time.monotonic() - last_status >= options.get(
                    "heartbeat_seconds", 30
                ):
                    print(
                        "[진행] verl 실행 중 · 상세 진행/오류는 train.log에 누적",
                        flush=True,
                    )
                    last_status = time.monotonic()
                time.sleep(options.get("poll_seconds", 1))
            if process.returncode:
                if "math_reasoning" in cfg:
                    write_json(
                        run / "result.json",
                        {
                            "completed": False,
                            "returncode": process.returncode,
                            "log": str(run / "train.log"),
                            "phase": phase,
                        },
                    )
                with (run / "train.log").open("rb") as failure_log:
                    failure_log.seek(max(0, failure_log.seek(0, 2) - 8000))
                    print(failure_log.read().decode(errors="replace"), file=sys.stderr)
                raise RuntimeError(
                    f"verl 종료 코드 {process.returncode}; {run / 'train.log'}"
                )
        finally:
            from scripts.utils.process import stop_group

            stop_group(process)
    if "math_reasoning" in cfg and phase != "test":
        artifact = json.loads((run / "checkpoint_export.json").read_text())
        if artifact["global_step"] != total_steps:
            raise RuntimeError(
                "Native worker exited without the requested final checkpoint"
            )
        write_json(run / "result.json", {**artifact, "completed": True})
    print("[완료] " + str(checkpoints / "last"), flush=True)


def legacy_main():
    parser = argparse.ArgumentParser(
        description="Native verl: start / resume / check / logs / status / test"
    )
    parser.add_argument(
        "command",
        choices=["start", "resume", "check", "logs", "status", "test", "help"],
    )
    parser.add_argument(
        "--seed", type=int, help="YAML의 launcher.seed 대신 사용할 seed"
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="새 실행의 설정 YAML"
    )
    parser.add_argument("--run", type=Path, help="특정 verl artifact run을 선택")
    args = parser.parse_args()
    if args.command == "help":
        parser.print_help()
        return
    # An explicit saved run remains usable even if today's YAML has been edited.
    settings = None
    if args.run is None or args.command in ("start", "check"):
        settings = load_settings(args.config, args.seed)
    run_command(args.command, settings, args.run)


def run_command(command, settings=None, selected_run=None):
    """Shared lifecycle; CLI choice does not change training/checkpoint semantics."""
    seed = settings.launcher.seed if settings is not None else None
    runs = Path(settings.launcher.artifacts_dir) if settings is not None else None
    if command in ("logs", "status"):
        run = (selected_run or latest_run(seed, runs)).resolve()
        manifest = read_manifest(run)
        if command == "logs":
            os.execvp("tail", ["tail", "-n", "50", "-F", str(run / "train.log")])
        print("RUN", run)
        last = Path(manifest["checkpoints"]) / "last"
        print("LAST", last.resolve() if last.is_symlink() else "아직 저장 없음")
        return
    # The controller validates on CPU. Only run_worker receives the selected GPUs.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from omegaconf import OmegaConf

    # One shared launcher lock also covers runs in different experiment directories.
    RUNS.mkdir(parents=True, exist_ok=True)
    with (RUNS / ".launcher.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("다른 verl 실행기가 이미 실행 중입니다.") from None
        if command in ("start", "check"):
            if selected_run:
                raise ValueError(
                    "start/check는 --run을 받지 않습니다. 새 run을 자동 생성합니다."
                )
            name = (
                datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S")
                + "-"
                + uuid.uuid4().hex[:8]
            )
            if command == "check":
                import tempfile

                with tempfile.TemporaryDirectory(
                    prefix="verl-config-check-"
                ) as temporary:
                    cfg, manifest = prepare_run(
                        Path(temporary) / name, settings=settings
                    )
                    check_environment(cfg)
                    batch = cfg.data.train_batch_size
                    generations = cfg.actor_rollout_ref.rollout.n
                    microbatch = (
                        cfg.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
                    )
                    accumulation = (
                        batch
                        * generations
                        // (cfg.trainer.n_gpus_per_node * microbatch)
                    )
                    print(
                        f"[설정] {cfg.trainer.total_training_steps} updates · {batch} prompts × G{generations} · "
                        f"microbatch{microbatch} × accum{accumulation} · "
                        f"{cfg.data.max_prompt_length}+{cfg.data.max_response_length}"
                    )
                return
            gpu_preflight(",".join(map(str, settings.launcher.gpu_ids)))
            run = runs / f"seed-{seed:04d}" / name
            cfg, manifest = prepare_run(run, settings=settings)
            check_environment(cfg)
            write_json(run.parent / "latest.json", {"run": str(run)})
        else:
            run = (selected_run or latest_run(seed, runs)).resolve()
            manifest = read_manifest(run)
            if "provenance" in manifest:
                from scripts.utils.manifest import (
                    validate_model_manifest,
                    validate_provenance,
                )

                validate_provenance(manifest["provenance"])
                validate_model_manifest(manifest["initialization"])
                initial_adapter = manifest.get("adapter_initialization")
                if initial_adapter:
                    from scripts.utils.io import sha256

                    for name, expected in initial_adapter["weight_sha256"].items():
                        if sha256(Path(initial_adapter["path"]) / name) != expected:
                            raise RuntimeError("Frozen initialization adapter changed")
            # Resume uses the frozen run settings, never today's experiment YAML.
            cfg = OmegaConf.load(run / "config.yaml")
            validate_launcher_config(cfg)
            world_size = cfg.trainer.n_gpus_per_node
            total_steps = cfg.trainer.total_training_steps
            for name, expected in manifest["parquet_sha256"].items():
                if (
                    hashlib.sha256((run / f"{name}.parquet").read_bytes()).hexdigest()
                    != expected
                ):
                    raise RuntimeError(f"동결 데이터 변경: {name}")
            root = Path(manifest["checkpoints"])
            publish_last(
                root, world_size, cfg.get("launcher", {}).get("keep_last_only", True)
            )
            last = root / "last"
            if not last.is_symlink() or last.resolve().parent != root.resolve():
                raise RuntimeError(f"재개할 verl last checkpoint가 없습니다: {last}")
            checkpoint = last.resolve()
            step = int(checkpoint.name.removeprefix("global_step_"))
            validate_checkpoint(root, step, world_size)
            if command == "resume" and step >= total_steps:
                print(
                    f"이미 {total_steps} updates를 완료했습니다. 최종 평가는 test 명령을 사용하세요."
                )
                return
            if command == "test" and step != total_steps:
                raise RuntimeError(
                    f"test는 고정된 최종 update {total_steps}에서만 실행합니다."
                )
            cfg.trainer.resume_mode = "resume_path"
            cfg.trainer.resume_from_path = str(checkpoint)
            if command == "test":
                cfg.data.val_files = [str(run / "test.parquet")]
                cfg.trainer.val_before_train = True
                cfg.trainer.val_only = True
                cfg.trainer.validation_data_dir = str(run / "test")
                cfg.trainer.experiment_name = run.name + "-test"
            if os.environ.get("WANDB_MODE") == "disabled":
                cfg.trainer.logger = ["file"]
            if "launcher" in cfg and "WANDB_MODE" in os.environ:
                cfg.launcher.wandb.mode = os.environ["WANDB_MODE"]
            check_environment(cfg)
            gpu_preflight(manifest["gpu_ids"])
        execute(run, cfg, manifest, command)
        return run


@hydra.main(
    version_base="1.3", config_path=str(REPO / "configs"), config_name="verl_math"
)
def hydra_main(settings):
    """Native Hydra CLI: config selection, overrides, inspection and serial sweeps."""
    from hydra.core.hydra_config import HydraConfig
    from hydra.core.override_parser.overrides_parser import OverridesParser

    command = settings.launcher.command
    if command not in {"start", "resume", "check", "logs", "status", "test"}:
        raise ValueError(f"알 수 없는 launcher.command: {command}")
    if command not in {"start", "check"}:
        # A resumed model must use its frozen config. Do not silently ignore new
        # training overrides that the caller might believe were applied.
        allowed = {
            "launcher.command",
            "launcher.run",
            "launcher.seed",
            "launcher.experiment",
            "launcher.base_dir",
            "launcher.artifacts_dir",
        }
        overrides = OverridesParser.create().parse_overrides(
            HydraConfig.get().overrides.task
        )
        changed = [
            item.key_or_group for item in overrides if item.key_or_group not in allowed
        ]
        if changed:
            raise ValueError(
                "resume/test/logs/status는 저장된 설정을 사용합니다. 새 학습 옵션을 제거하세요: "
                + ", ".join(changed)
            )
    selected_run = Path(settings.launcher.run) if settings.launcher.run else None
    run_command(command, settings, selected_run)


def main():
    # Keep the established start/resume/logs commands. All other invocations,
    # including no arguments, use Hydra's actual CLI rather than a custom parser.
    if len(sys.argv) > 1 and sys.argv[1] in {
        "start",
        "resume",
        "check",
        "logs",
        "status",
        "test",
        "help",
    }:
        legacy_main()
    else:
        register_resolvers()
        hydra_main()
