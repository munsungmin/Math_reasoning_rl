"""Native verl launcher, reward bridge and rolling checkpoint publication.

User entry point: run_math_verl.sh. No GPU initialization occurs on import.
Settings live in configs/verl_math.yaml; each run freezes its effective config.
"""

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import hydra
from hydra.core.config_search_path import ConfigSearchPath
from hydra.core.plugins import Plugins
from hydra.plugins.search_path_plugin import SearchPathPlugin

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs/verl_math.yaml"
# Defaults below are only for compatibility with runs saved before YAML extraction.
BASE = Path("/data/sungmin/math_reasoning")
RUNS = BASE / "artifacts/math_error_transfer/verl/E1"
CHECKPOINTS = BASE / "checkpoints/math_error_transfer/verl/E1"
GRADER_PY = BASE / "envs/limit-rlvr-grader/bin/python"
WORLD_SIZE = 4


class VerlConfigSearchPath(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath):
        search_path.append(
            provider="verl", path=f"file://{REPO / 'ext/verl/verl/trainer/config'}"
        )


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    *,
    grader_python=GRADER_PY,
    grader_script=REPO / "script/limit_rlvr_grade.py",
    grader_timeout=90,
    correctness_weight=2.0,
    format_weight=0.5,
    numeric_weight=0.5,
    **kwargs,
):
    """Same symbolic correctness, format and numeric rewards as the verified probe."""
    result = subprocess.run(
        [str(grader_python), str(grader_script)],
        input=json.dumps(
            [
                {
                    "dataset": "math500_level3_5",
                    "completion": solution_str,
                    "answer": ground_truth,
                }
            ]
        ),
        text=True,
        capture_output=True,
        timeout=grader_timeout,
        check=False,
    )
    if result.returncode:
        raise RuntimeError("MATH grader failed: " + result.stderr[-4000:])
    grade = json.loads(result.stdout)[0]
    correct = float(grade["correct"])
    format_score = float(
        bool(
            re.search(
                r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>",
                solution_str,
                re.DOTALL,
            )
        )
    )
    numeric_score = 0.0
    if "<answer>" in solution_str:
        # Preserve the public reward's last-tag extraction, including an open tag.
        numeric_answer = solution_str.split("<answer>")[-1].split("</answer>")[0]
        try:
            float(numeric_answer.replace(",", "").strip())
            numeric_score = 1.0
        except ValueError:
            pass
    return {
        "score": correctness_weight * correct
        + format_weight * format_score
        + numeric_weight * numeric_score,
        "acc": correct,
        "format": format_score,
        "numeric": numeric_score,
    }


def register_resolvers():
    """Small, explicit Hydra interpolations; no CUDA imports or code evaluation."""
    from omegaconf import OmegaConf

    # Register outside YAML so experiment configs can inherit verl_math.
    if VerlConfigSearchPath not in Plugins.instance().discover(SearchPathPlugin):
        Plugins.instance().register(VerlConfigSearchPath)
    OmegaConf.register_new_resolver("repo_root", lambda: str(REPO), replace=True)
    OmegaConf.register_new_resolver("token_budget", lambda a, b: a + b, replace=True)
    OmegaConf.register_new_resolver("gpu_count", len, replace=True)
    OmegaConf.register_new_resolver(
        "rollout_count", lambda steps, prompts, samples: steps * prompts * samples,
        replace=True,
    )


def load_settings(path=DEFAULT_CONFIG, seed=None):
    """Compose the same full Hydra config for compatibility commands and tests."""
    register_resolvers()
    path = Path(path).resolve()
    with hydra.initialize_config_dir(config_dir=str(path.parent), version_base="1.3"):
        settings = hydra.compose(config_name=path.name)
    if seed is not None:
        settings.launcher.seed = seed
    return settings


def build_config(run, checkpoints=None, seed=None, settings=None):
    """Compose upstream Hydra defaults and the experiment YAML, then freeze values."""
    from omegaconf import OmegaConf

    settings = OmegaConf.create(settings) if settings is not None else load_settings()
    if seed is not None:
        settings.launcher.seed = seed
    if checkpoints is None:
        checkpoints = (
            Path(settings.launcher.checkpoints_dir)
            / f"seed-{settings.launcher.seed:04d}"
            / run.name
        )
    settings.runtime.update(
        run_dir=str(run), run_name=run.name, checkpoint_dir=str(checkpoints)
    )
    # Both entry points already receive the fully composed Hydra configuration.
    cfg = settings
    OmegaConf.resolve(cfg)
    if Path(cfg.trainer.default_local_dir) != Path(checkpoints):
        raise ValueError("저장 위치는 launcher.checkpoints_dir에서 변경하세요.")
    if cfg.trainer.resume_mode != "disable" or cfg.trainer.val_only:
        raise ValueError("재개/최종 평가는 YAML 대신 resume/test 명령으로 선택하세요.")
    if cfg.launcher.wandb.mode == "disabled":
        cfg.trainer.logger = [name for name in cfg.trainer.logger if name != "wandb"]
    validate_launcher_config(cfg)
    return cfg


def validate_launcher_config(cfg):
    """Check relationships needed for one update per step and safe native resume."""
    actor, rollout = cfg.actor_rollout_ref.actor, cfg.actor_rollout_ref.rollout
    if cfg.actor_rollout_ref.model.lora_rank <= 0:
        raise ValueError("현재 checkpoint 검증 경로는 LoRA 학습용입니다.")
    if "launcher" in cfg:
        ids = cfg.launcher.gpu_ids
        if not ids or len(set(ids)) != len(ids) or not set(ids) <= {2, 3, 4, 5, 6}:
            raise ValueError(
                "launcher.gpu_ids에는 사용 가능한 GPU 2~6을 중복 없이 지정하세요."
            )
        if cfg.trainer.n_gpus_per_node != len(ids):
            raise ValueError("GPU 개수와 trainer.n_gpus_per_node가 다릅니다.")
        if cfg.launcher.wandb.mode not in ("online", "offline", "disabled"):
            raise ValueError("W&B mode는 online/offline/disabled 중 하나여야 합니다.")
        if min(cfg.launcher.poll_seconds, cfg.launcher.heartbeat_seconds) <= 0:
            raise ValueError("로그 갱신 주기는 양수여야 합니다.")
        for key in ("artifacts_dir", "checkpoints_dir"):
            if not Path(cfg.launcher[key]).resolve().is_relative_to("/data"):
                raise ValueError(f"launcher.{key}는 /data 아래에 지정하세요.")
    if cfg.trainer.nnodes != 1 or cfg.trainer.use_v1 or actor.strategy != "fsdp":
        raise ValueError(
            "이 실행기는 현재 단일 노드 native FSDP/use_v1=false 경로용입니다."
        )
    if cfg.algorithm.adv_estimator != "grpo" or rollout.n < 2:
        raise ValueError("현재 실행기는 GRPO, rollout.n >= 2 설정을 사용합니다.")
    if actor.ppo_epochs != 1 or actor.ppo_mini_batch_size != cfg.data.train_batch_size:
        raise ValueError(
            "1 step = 1 update를 위해 ppo_epochs=1, mini_batch=train_batch를 유지하세요."
        )
    if min(cfg.data.train_batch_size, actor.ppo_micro_batch_size_per_gpu) <= 0:
        raise ValueError("학습 batch와 microbatch는 양수여야 합니다.")
    if actor.use_dynamic_bsz or actor.fsdp_config.ulysses_sequence_parallel_size != 1:
        raise ValueError(
            "이 실행기의 batch 검사는 고정 microbatch, sequence parallel=1 기준입니다."
        )
    trajectories = cfg.data.train_batch_size * rollout.n
    if trajectories % (
        cfg.trainer.n_gpus_per_node * actor.ppo_micro_batch_size_per_gpu
    ):
        raise ValueError(
            "trajectory batch가 GPU 개수 × microbatch로 나누어떨어져야 합니다."
        )
    if min(cfg.data.max_prompt_length, cfg.data.max_response_length) <= 0:
        raise ValueError("prompt/response 길이는 양수여야 합니다.")
    if (
        rollout.max_model_len
        < cfg.data.max_prompt_length + cfg.data.max_response_length
    ):
        raise ValueError("max_model_len이 prompt + response 길이보다 작습니다.")
    if (
        rollout.max_num_batched_tokens < rollout.max_model_len
        and not rollout.enable_chunked_prefill
    ):
        raise ValueError(
            "chunked prefill을 끄면 batched tokens >= max_model_len이 필요합니다."
        )
    if min(cfg.trainer.total_training_steps, cfg.trainer.save_freq) <= 0:
        raise ValueError("목표 updates와 저장 주기는 양수여야 합니다.")
    if "file" not in cfg.trainer.logger:
        raise ValueError(
            "누적 로그와 checkpoint 감시를 위해 trainer.logger에 file이 필요합니다."
        )
    if (
        cfg.trainer.max_actor_ckpt_to_keep is not None
        or cfg.trainer.del_local_ckpt_after_load
    ):
        raise ValueError(
            "native checkpoint 선삭제를 끄고 launcher.keep_last_only를 사용하세요."
        )
    required = {"model", "optimizer", "extra"}
    if (
        not required <= set(actor.checkpoint.save_contents)
        or not required <= set(actor.checkpoint.load_contents)
        or actor.checkpoint.async_save
    ):
        raise ValueError(
            "last 재개에는 model/optimizer/extra의 동기 저장·복원이 필요합니다."
        )
    if actor.optim.total_training_steps != cfg.trainer.total_training_steps:
        raise ValueError("optimizer와 trainer의 total_training_steps를 일치시키세요.")


def write_json(path, value):
    temporary = path.with_name(f".{path.name}-{uuid.uuid4().hex}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


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


def read_rows(path):
    return [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]


def check_disjoint(*splits):
    """Reject duplicate IDs, families, or normalized problem text across splits."""
    for field in ("sample_id", "family_id", "problem"):
        seen = set()
        for rows in splits:
            keys = [str(row[field]) for row in rows]
            if field == "problem":
                keys = ["".join(key.lower().split()) for key in keys]
            if len(set(keys)) != len(keys) or seen.intersection(keys):
                raise ValueError(f"Duplicate/overlapping {field} in evaluation splits")
            seen.update(keys)


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
        counts != dict(cfg.launcher.dataset.expected_counts)
        or min(counts.values()) <= 0
    ):
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
    for name, rows in splits.items():
        records = []
        for index, row in enumerate(rows):
            if row["level"] not in cfg.launcher.dataset.levels:
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
    write_json(run / "manifest.json", manifest)
    return cfg, manifest


def validate_checkpoint(root, step, world_size=WORLD_SIZE):
    """The native completion marker is written only after all ranks finish saving."""
    checkpoint = root / f"global_step_{step}"
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise RuntimeError(f"실제 checkpoint 폴더가 아닙니다: {checkpoint}")
    required = [
        "data.pt",
        "actor/fsdp_config.json",
        "actor/lora_train_meta.json",
        "actor/huggingface/config.json",
        "actor/huggingface/tokenizer_config.json",
    ]
    required += [
        f"actor/{kind}_world_size_{world_size}_rank_{rank}.pt"
        for rank in range(world_size)
        for kind in ("model", "optim", "extra_state")
    ]
    for name in required:
        path = checkpoint / name
        if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
            raise RuntimeError(f"불완전한 checkpoint: {path}")
    fsdp_config = json.loads((checkpoint / "actor/fsdp_config.json").read_text())
    if fsdp_config["world_size"] != world_size:
        raise RuntimeError(f"Checkpoint GPU world size가 {world_size}가 아닙니다.")
    return checkpoint


def publish_last(root, world_size=WORLD_SIZE, keep_last_only=True):
    """Publish the complete save first; delete only older saves in this run."""
    marker = root / "latest_checkpointed_iteration.txt"
    if not marker.exists():
        return None
    value = marker.read_text().strip()
    if not value:  # Native writer briefly truncates the marker before writing.
        return None
    step = int(value)
    last = root / "last"
    if last.exists() and not last.is_symlink():
        raise RuntimeError(f"last가 symlink가 아니므로 교체하지 않습니다: {last}")
    if last.is_symlink() and last.resolve().parent != root.resolve():
        raise RuntimeError("last가 현재 실행 밖을 가리킵니다.")
    checkpoint = validate_checkpoint(root, step, world_size)
    changed = not last.is_symlink() or last.resolve() != checkpoint.resolve()
    if changed:
        temporary = root / f".last-{uuid.uuid4().hex}"
        try:
            temporary.symlink_to(checkpoint.name, target_is_directory=True)
            os.replace(temporary, last)
        finally:
            temporary.unlink(missing_ok=True)
    for previous in root.iterdir():
        match = re.fullmatch(r"global_step_(\d+)", previous.name)
        if (
            keep_last_only
            and match
            and int(match[1]) < step
            and previous.is_dir()
            and not previous.is_symlink()
        ):
            shutil.rmtree(previous)
    return step if changed else None


def latest_run(seed, runs=RUNS):
    pointer = runs / f"seed-{seed:04d}" / "latest.json"
    return Path(json.loads(pointer.read_text())["run"])


def gpu_preflight(gpus):
    """Read NVIDIA process lists; never stop an existing GPU job."""

    def query(flag):
        return subprocess.run(
            ["nvidia-smi", flag, "--format=csv,noheader,nounits"],
            check=True,
            text=True,
            capture_output=True,
            timeout=10,
        ).stdout

    devices = {
        uuid.strip(): index.strip()
        for line in query("--query-gpu=index,uuid").splitlines()
        for index, uuid in [line.split(",")]
    }
    selected = set(gpus.split(","))
    if not selected.issubset(set(devices.values())):
        raise RuntimeError(f"필요한 GPU를 찾을 수 없습니다: {gpus}")
    busy = [
        line
        for line in query("--query-compute-apps=gpu_uuid,pid").splitlines()
        if devices.get(line.split(",")[0].strip()) in selected
    ]
    if busy:
        raise RuntimeError(f"GPU {gpus}에 기존 작업이 있어 시작하지 않습니다: {busy}")


def check_environment(cfg):
    """CPU-only import/config/reward validation; never launches Ray or training."""
    import verl
    from verl.utils.config import validate_config

    if Path(verl.__file__).resolve().parent != REPO / "ext/verl/verl":
        raise RuntimeError(f"잘못된 verl 설치 경로입니다: {verl.__file__}")
    if (
        "wandb" in cfg.trainer.logger
        and cfg.get("launcher", {})
        .get("wandb", {})
        .get("mode", os.environ.get("WANDB_MODE", "online"))
        == "online"
    ):
        import wandb

        if not wandb.login(prompt=False, verify=False):
            raise RuntimeError(
                "W&B 로그인 정보가 없습니다. verl-current 환경에서 wandb login을 하거나 WANDB_MODE=offline으로 실행하세요."
            )
    model = Path(cfg.actor_rollout_ref.model.path)
    if not model.is_dir():
        raise FileNotFoundError(f"로컬 모델이 없습니다: {model}")
    validate_launcher_config(cfg)
    validate_config(
        cfg,
        use_reference_policy=cfg.actor_rollout_ref.actor.use_kl_loss
        or cfg.algorithm.use_kl_in_reward,
        use_critic=False,
    )
    from verl.trainer.ppo.reward import get_custom_reward_fn

    reward_fn = get_custom_reward_fn(cfg)
    for answer, expected in [("9", 1.0), ("8", 0.0)]:
        if reward_fn("math500", f"<answer>{answer}</answer>", "9")["acc"] != expected:
            raise RuntimeError("MATH 판정기 사전 검사 실패")
    print("[검사] ext/verl · 로컬 Qwen-Math · 설정 · MATH 판정기 통과", flush=True)


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


def run_worker(config_path):
    """Only this explicitly launched child imports CUDA-facing trainer modules."""
    import ray
    from omegaconf import OmegaConf
    from verl.trainer.main_ppo import main as verl_main

    def stop_worker(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_worker)
    try:
        verl_main(OmegaConf.load(config_path))
    finally:
        ray.shutdown()


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
            [sys.executable, str(Path(__file__).resolve()), "worker", str(config_path)],
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
                published = publish_last(checkpoints, world_size, keep_last_only)
                if published is not None:
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
                with (run / "train.log").open("rb") as failure_log:
                    failure_log.seek(max(0, failure_log.seek(0, 2) - 8000))
                    print(failure_log.read().decode(errors="replace"), file=sys.stderr)
                raise RuntimeError(
                    f"verl 종료 코드 {process.returncode}; {run / 'train.log'}"
                )
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
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


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "worker":
        run_worker(sys.argv[2])
    else:
        main()
