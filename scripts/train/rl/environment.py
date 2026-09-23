"""Extracted native verl environment; legacy run semantics are preserved."""

import os
import subprocess
from pathlib import Path

from scripts.utils.paths import REPO

from .native_config import validate_launcher_config


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
