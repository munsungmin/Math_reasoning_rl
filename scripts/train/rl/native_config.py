"""Extracted native verl native config; legacy run semantics are preserved."""

from pathlib import Path

from scripts.utils.paths import REPO

DEFAULT_CONFIG = REPO / "configs/verl_math.yaml"
import hydra
from hydra.core.config_search_path import ConfigSearchPath
from hydra.core.plugins import Plugins
from hydra.plugins.search_path_plugin import SearchPathPlugin


class VerlConfigSearchPath(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath):
        search_path.append(
            provider="verl", path=f"file://{REPO / 'ext/verl/verl/trainer/config'}"
        )


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
        "rollout_count",
        lambda steps, prompts, samples: steps * prompts * samples,
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
