#!/usr/bin/env bash
# Qwen2.5-1.5B / GSM8K / Megatron GRPO. conda activate verl-megatron
# Current verl Megatron code requires BF16 + FlashAttention 2 (Ampere or newer).
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT/verl-cu121"
export CUDA_VISIBLE_DEVICES=3,4,5,6
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
INSPECT_CONFIG=0
for arg in "$@"; do
  case "$arg" in --cfg|--cfg=*|--help|-h) INSPECT_CONFIG=1 ;; esac
done
if [[ "$INSPECT_CONFIG" == 0 ]]; then
  python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit('CUDA is unavailable in this environment.')
unsupported = [(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())
               if torch.cuda.get_device_capability(i)[0] < 8]
if unsupported:
    raise SystemExit('This verl Megatron backend requires BF16 and FlashAttention 2. '
                     f'Unsupported GPUs: {unsupported}. Use script/run_fsdp.sh on RTX 2080 Ti.')
import megatron.core, apex, flash_attn
PY
fi
python -m verl.trainer.main_ppo --config-name=ppo_megatron_trainer \
  algorithm.adv_estimator=grpo \
  data.train_files="${TRAIN_FILE:-$TASK_ROOT/dataset/gsm8k/train.parquet}" \
  data.val_files="${VAL_FILE:-$TASK_ROOT/dataset/gsm8k/test.parquet}" \
  data.filter_overlong_prompts=true \
  data.train_batch_size=4 data.max_prompt_length=256 data.max_response_length=128 \
  actor_rollout_ref.model.path="${MODEL_PATH:-Qwen/Qwen2.5-1.5B}" \
  actor_rollout_ref.actor.strategy=megatron \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_torch_compile=false \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2 \
  actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=2 \
  actor_rollout_ref.actor.megatron.sequence_parallel=true \
  actor_rollout_ref.ref.megatron.tensor_model_parallel_size=2 \
  actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=2 \
  actor_rollout_ref.ref.megatron.sequence_parallel=true \
  actor_rollout_ref.ref.param_offload=true \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.rollout.enable_chunked_prefill=false \
  actor_rollout_ref.rollout.max_num_batched_tokens=512 \
  actor_rollout_ref.rollout.max_num_seqs=2 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.n=2 \
  critic.megatron.tensor_model_parallel_size=2 critic.megatron.pipeline_model_parallel_size=2 \
  trainer.n_gpus_per_node=4 trainer.nnodes=1 \
  trainer.logger=console trainer.project_name=qwen25-math \
  trainer.experiment_name=grpo-megatron +trainer.val_before_train=false \
  trainer.save_freq=-1 trainer.test_freq=-1 trainer.total_epochs=1 \
  "$@"
