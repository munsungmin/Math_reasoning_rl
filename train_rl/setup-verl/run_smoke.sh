#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
export VERL_ATTN_IMPLEMENTATION=sdpa
export VLLM_ATTENTION_BACKEND=XFORMERS
export OMP_NUM_THREADS=4
python -m pip check
CUDA_VISIBLE_DEVICES='' python script/check_datasets.py
CUDA_VISIBLE_DEVICES='' python setup-verl/smoke_ray.py
CUDA_VISIBLE_DEVICES=3,4,5,6 python setup-verl/smoke_test.py
CUDA_VISIBLE_DEVICES=3 python setup-verl/smoke_qwen.py
CUDA_VISIBLE_DEVICES=3,4,5,6 torchrun --standalone --nproc_per_node=4 setup-verl/smoke_nccl.py
bash script/run_fsdp.sh --cfg job > setup-verl/grpo-resolved.yaml
