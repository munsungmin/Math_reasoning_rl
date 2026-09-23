"""Compile independently selected project components into a native verl config."""

from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict

from scripts.reasoning.model import ModelSpec
from scripts.utils.config import native_defaults, plain
from scripts.utils.paths import REPO


def to_native(project, *, resolve_model=True):
    p = plain(project)
    model, adapter = p["model"], p["model"]["adapter"]
    if adapter["rank"] <= 0 or adapter["dropout"] != 0:
        raise ValueError(
            "Current verl FSDP training requires LoRA rank > 0 and dropout = 0"
        )
    if p["optimizer"]["_target_"] != "torch.optim.AdamW":
        raise ValueError("This native optimizer adapter supports torch.optim.AdamW")
    if p["lr_scheduler"]["name"] not in ("constant", "cosine"):
        raise ValueError("Current verl scheduler supports constant/cosine")
    if p["task"]["kind"] != "rl":
        raise ValueError("Native RL requires task=rl")
    for name, observer in p["callbacks"].items():
        interval = observer["every_n_steps"]
        if interval % p["trainer"]["save_every"]:
            raise ValueError(
                f"RL observation {name} requires an interval divisible by save_every"
            )
        if observer["_target_"].endswith("WeightNorm") and not observer.get(
            "trainable_only", True
        ):
            raise ValueError(
                "RL checkpoint observations currently expose adapter parameters only"
            )
    sampling, evaluation = (
        instantiate(project.rollout),
        instantiate(project.test_config),
    )
    from scripts.reasoning.rollout.sampling import NativeValidationSampling

    if not isinstance(evaluation, NativeValidationSampling):
        raise ValueError(
            "RL validation uses test_config=native_greedy/native_sample32; independent evaluation uses greedy/sample32"
        )
    objective, advantage = instantiate(project.loss), instantiate(project.advantage)
    cfg = native_defaults()
    backend = p["backend"]
    runtime, training, trainer = p["runtime"], p["train_config"], p["trainer"]
    base = runtime["base_dir"]
    experiment = p["experiment_name"]
    initialization = model["initialization"]
    if resolve_model:
        spec = ModelSpec.resolve(initialization)
        spec.validate_lora(adapter)
        spec.validate_native_tokenizer()
        path, adapter_path = spec.path, spec.adapter
        # Native verl uses one tokenizer with its model; mismatching tokenizers are not silently accepted.
        if initialization.get("tokenizer") and spec.tokenizer != spec.path:
            raise ValueError(
                "Native RL tokenizer overrides require a model artifact containing that tokenizer"
            )
    else:
        path, adapter_path = initialization["path"], initialization.get("adapter")
    with open_dict(cfg):
        for name in ("data", "trainer", "ray_kwargs"):
            cfg[name] = OmegaConf.merge(cfg[name], backend.get(name, {}))
        for name in ("actor", "model", "rollout", "ref"):
            cfg.actor_rollout_ref[name] = OmegaConf.merge(
                cfg.actor_rollout_ref[name], backend[name]
            )
        cfg.runtime = {
            "repo_dir": str(REPO),
            "run_name": "__generated_by_launcher__",
            "run_dir": "__generated_by_launcher__",
            "checkpoint_dir": "__generated_by_launcher__",
        }
        cfg.launcher = {
            "command": "start",
            "run": None,
            "seed": p["seed"],
            "experiment": experiment,
            "base_dir": base,
            "artifacts_dir": runtime["output_dir"]
            or f"{base}/artifacts/composable/{experiment}",
            "checkpoints_dir": f"{base}/checkpoints/composable/{experiment}",
            "gpu_ids": runtime["gpu_ids"],
            "dataset": {
                key: p["data"][key]
                for key in ("name", "sources", "expected_counts", "levels")
            },
            "wandb": {"mode": p["logger"]["mode"], "entity": p["logger"]["entity"]},
            "keep_last_only": trainer["keep_last_only"],
            "heartbeat_seconds": trainer["heartbeat_seconds"],
            "poll_seconds": trainer["poll_seconds"],
            "worker_env": runtime["worker_env"],
        }
        cfg.data.update(
            train_files=["${runtime.run_dir}/train.parquet"],
            val_files=["${runtime.run_dir}/validation.parquet"],
            train_batch_size=training["batch_size"],
            seed=p["data_seed"],
            dataloader_num_workers=runtime["num_workers"],
            train_max_samples=p["data"]["max_samples"],
            max_prompt_length=p["data"]["prompt"]["max_prompt_length"],
            max_response_length=sampling.max_new_tokens,
            rollout_distinct_question=training["batch_size"],
            rollout_per_question=sampling.n,
            backprop_rollout_per_gpu=training["micro_batch_size"],
            rollout_gpu=len(runtime["gpu_ids"]),
        )
        cfg.actor_rollout_ref.hybrid_engine = True
        cfg.actor_rollout_ref.nccl_timeout = 600
        cfg.actor_rollout_ref.model.update(
            path=path,
            lora_adapter_path=adapter_path,
            custom_chat_template=p["data"]["prompt"]["chat_template"],
            override_config={"attn_implementation": model["attention"]},
            enable_gradient_checkpointing=model["gradient_checkpointing"],
            lora_rank=adapter["rank"],
            lora_alpha=adapter["alpha"],
            target_modules=adapter["target_modules"],
        )
        actor = cfg.actor_rollout_ref.actor
        cfg.actor_rollout_ref.actor = OmegaConf.merge(actor, objective.native())
        actor = cfg.actor_rollout_ref.actor
        actor.update(
            ppo_mini_batch_size=training["batch_size"],
            ppo_micro_batch_size_per_gpu=training["micro_batch_size"],
            ppo_epochs=training["ppo_epochs"],
            shuffle=training["actor_shuffle"],
            data_loader_seed=training["actor_data_seed"],
            grad_clip=training["grad_clip"],
        )
        optimizer, scheduler = p["optimizer"], p["lr_scheduler"]
        actor.optim.update(
            optimizer="AdamW",
            optimizer_impl="torch.optim",
            lr=optimizer["lr"],
            weight_decay=optimizer["weight_decay"],
            betas=optimizer["betas"],
            lr_scheduler_type=scheduler["name"],
            lr_warmup_steps=scheduler["warmup_steps"],
            lr_warmup_steps_ratio=scheduler["warmup_ratio"],
            total_training_steps=training["max_steps"],
            clip_grad=training["grad_clip"],
            min_lr_ratio=scheduler["min_lr_ratio"],
            num_cycles=scheduler["num_cycles"],
        )
        if optimizer["eps"] != 1e-8:
            actor.optim.override_optimizer_config = {"eps": optimizer["eps"]}
        rollout = cfg.actor_rollout_ref.rollout
        rollout.update(
            **sampling.native(),
            seed=sampling.seed,
            max_model_len=cfg.data.max_prompt_length + sampling.max_new_tokens,
            max_num_batched_tokens=cfg.data.max_prompt_length + sampling.max_new_tokens,
            val_kwargs=OmegaConf.merge(rollout.val_kwargs, evaluation.native()),
        )
        cfg.algorithm.update(**advantage.native(), use_kl_in_reward=False)
        grader = p["grader"]
        # Instantiate now to reject incompatible component/weight mappings before any GPU job.
        instantiate(project.reward)
        cfg.reward.num_workers = 1
        cfg.reward.reward_manager.source = "importlib"
        cfg.reward.reward_manager.name = "RecordedRewardManager"
        cfg.reward.reward_manager.module.path = str(
            REPO / "scripts/train/rl/reward_manager.py"
        )
        cfg.reward.custom_reward_function = {
            "path": str(REPO / "scripts/reasoning/reward/verl.py"),
            "name": "compute_score",
            "reward_kwargs": {
                "grader_python": grader["python"],
                "grader_script": grader["script"],
                "grader_timeout": grader["timeout"],
                "components": p["reward"]["components"],
                "weights": p["reward"]["weights"],
            },
        }
        cfg.trainer.update(
            n_gpus_per_node=len(runtime["gpu_ids"]),
            total_training_steps=training["max_steps"],
            total_epochs=training["max_epochs"],
            total_rollout=training["max_steps"] * training["batch_size"] * sampling.n,
            save_freq=trainer["save_every"],
            test_freq=trainer["validate_every"],
            val_before_train=trainer["validate_before_train"],
            logger=["file"] + (["wandb"] if p["logger"]["mode"] != "disabled" else []),
            project_name=p["logger"]["project"],
            experiment_name="${runtime.run_name}",
            default_local_dir="${runtime.checkpoint_dir}",
            rollout_data_dir="${runtime.run_dir}/rollouts",
            validation_data_dir="${runtime.run_dir}/validation",
        )
        cfg.math_reasoning = p
    return cfg
