#!/usr/bin/env bash
# Qwen2.5-1.5B / GSM8K / FSDP GRPO. Run after: conda activate verl
# Prepare parquet files with: python script/prepare_datasets.py
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
cd "$TASK_ROOT/verl-cu121"
export CUDA_VISIBLE_DEVICES=3,4,5,6
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export VERL_ATTN_IMPLEMENTATION=sdpa
export VLLM_ATTENTION_BACKEND=XFORMERS
export TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHDYNAMO_DISABLE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="${TRAIN_FILE:-$TASK_ROOT/dataset/gsm8k/train.parquet}" \
  data.val_files="${VAL_FILE:-$TASK_ROOT/dataset/gsm8k/test.parquet}" \
  data.filter_overlong_prompts=true \
  data.train_batch_size=4 data.max_prompt_length=256 data.max_response_length=128 \
  actor_rollout_ref.model.path="${MODEL_PATH:-Qwen/Qwen2.5-1.5B}" \
  actor_rollout_ref.model.use_remove_padding=false \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_torch_compile=false \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  +actor_rollout_ref.actor.fsdp_config.model_dtype=fp32 \
  +actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype=fp32 \
  +actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce_dtype=fp32 \
  +actor_rollout_ref.actor.fsdp_config.mixed_precision.buffer_dtype=fp32 \
  actor_rollout_ref.ref.fsdp_config.param_offload=true \
  +actor_rollout_ref.ref.fsdp_config.model_dtype=fp32 \
  +actor_rollout_ref.ref.fsdp_config.mixed_precision.param_dtype=fp32 \
  +actor_rollout_ref.ref.fsdp_config.mixed_precision.reduce_dtype=fp32 \
  +actor_rollout_ref.ref.fsdp_config.mixed_precision.buffer_dtype=fp32 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.dtype=float16 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.rollout.enable_chunked_prefill=false \
  actor_rollout_ref.rollout.max_num_batched_tokens=512 \
  actor_rollout_ref.rollout.max_num_seqs=2 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.n=2 \
  trainer.n_gpus_per_node=4 trainer.nnodes=1 \
  trainer.logger=console trainer.project_name=qwen25-math \
  trainer.experiment_name=grpo-2080ti +trainer.val_before_train=false \
  trainer.default_local_dir="$TASK_ROOT/outputs/checkpoints/qwen25-math/grpo-2080ti" \
  hydra.run.dir="$TASK_ROOT/outputs/hydra/fsdp/$RUN_ID" \
  trainer.save_freq=-1 trainer.test_freq=-1 trainer.total_epochs=1 \
  "$@"
