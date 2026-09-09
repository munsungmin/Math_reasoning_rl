# Qwen2.5-1.5B용 verl 환경 (RTX 2080 Ti)

## 환경 사용

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate verl
cd /home/sungmin/math_reasoning/verl-cu121
```

- Conda: `/home/sungmin/.conda/envs/verl`, Python 3.10
- 설치 소스: `verl-cu121`, `v0.3.0.post1` (`070ed6ac`), editable 설치
- 원래 `verl/` 클론은 변경하지 않음. `verl-cu121`은 원본 저장소의 git worktree이므로 원본 `.git`을 삭제하지 말 것.
- PyTorch 2.4.0 / CUDA runtime 12.1, vLLM 0.6.3, Transformers 4.45.2
- GPU: RTX 2080 Ti 11 GB × 7, NVIDIA driver 530.30.02
- Conda가 Python을 관리하고 uv pip가 해당 conda 환경 안에 Python 라이브러리를 설치함.

최신 클론의 CUDA 13.0 / torch 2.11 구성은 현재 드라이버와 맞지 않아 이전 릴리스를 사용했다. RTX 2080 Ti에서 FlashAttention 2와 BF16 기본 설정을 사용할 수 없으므로, 학습은 FP32 + SDPA, vLLM 생성은 FP16 + xFormers로 구성했다. FlashAttention, Megatron, SGLang은 이 구성에 필요하지 않다.

공식 자료:
- https://verl.readthedocs.io/en/latest/start/install.html
- https://docs.vllm.ai/en/v0.6.3/getting_started/installation.html
- https://github.com/Dao-AILab/flash-attention

## 로컬 호환 수정

`compatibility.patch`에 변경을 보관한다.

- FSDP 모델 로딩에서 `VERL_ATTN_IMPLEMENTATION=sdpa` 선택 허용.
- actor/critic의 FlashAttention padding import는 remove-padding 사용 시에만 실행.
- BF16 미지원 GPU에서는 actor/critic BF16 autocast를 끄고 모델의 FP32 설정을 사용.

`conda activate verl` 시 `CUDA_VISIBLE_DEVICES=3,4,5,6`, `NCCL_P2P_DISABLE=1`, `NCCL_IB_DISABLE=1`, `VERL_ATTN_IMPLEMENTATION=sdpa`, `VLLM_ATTENTION_BACKEND=XFORMERS`, `TOKENIZERS_PARALLELISM=false`, `OMP_NUM_THREADS=4`가 적용된다. `use_remove_padding=false`, sequence parallel size 1을 유지하고, BF16을 기본값으로 사용하는 다른 예제보다 제공된 런처를 사용한다.

## Smoke tests

```bash
cd /home/sungmin/math_reasoning
python -m pip check
CUDA_VISIBLE_DEVICES=3,4,5,6 python setup-verl/smoke_test.py
CUDA_VISIBLE_DEVICES="" python setup-verl/smoke_ray.py
CUDA_VISIBLE_DEVICES=3,4,5,6 torchrun --standalone --nproc_per_node=4 setup-verl/smoke_nccl.py
CUDA_VISIBLE_DEVICES=3 python setup-verl/smoke_qwen.py
```

- `smoke_test.py`: 라이브러리 및 verl/FSDP/vLLM 연결 모듈 import, 모든 GPU의 FP16 행렬 곱, 작은 랜덤 Qwen2 모델을 사용한 실제 verl actor forward/backward/optimizer.
- `smoke_ray.py`: 별도 CPU 프로세스에서 Ray remote task와 worker의 verl/PyTorch import.
- `smoke_nccl.py`: 사용 가능 GPU 3,4,5,6의 NCCL all-reduce.
- `smoke_qwen.py`: 실제 `Qwen/Qwen2.5-1.5B` 가중치를 Hugging Face cache에 다운로드하고 vLLM FP16 생성.
- 이 테스트는 전체 Qwen 1.5B GRPO 학습을 검증하는 end-to-end 테스트가 아니다.

## 학습 시작 예시

```bash
cd /home/sungmin/math_reasoning/verl-cu121
python ../script/prepare_datasets.py
cd ..
bash script/run_fsdp.sh trainer.total_training_steps=1
```

GSM8K 기본 경로는 `dataset/gsm8k/{train,test}.parquet`이며 256 토큰을 넘는 prompt는 로딩 시 필터링한다.

기본 모델은 사용자가 지정한 base 모델 `Qwen/Qwen2.5-1.5B`다. Instruct 모델을 쓰려면 `MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct`를 설정한다. 기본값은 GPU 3,4,5,6 (4 GPU), tensor parallel 1, GRPO n=2, 짧은 context, CPU offload다. GPU 개수를 바꾸면 train/mini batch의 divisibility도 맞춰야 한다. 데이터 전처리와 실제 전체 RL 학습은 별도 실행 단계이며, 런처의 메모리 사용량은 실제 데이터와 함께 조정해야 한다.

## 재설치

```bash
conda create -n verl python=3.10 pip -y
conda activate verl
python -m pip install uv
cd /home/sungmin/math_reasoning
# worktree가 없는 새 설치에서만:
git -C verl worktree add --detach ../verl-cu121 v0.3.0.post1
git -C verl-cu121 apply ../setup-verl/compatibility.patch
uv pip install --python "$CONDA_PREFIX/bin/python" -r setup-verl/requirements.in -e ./verl-cu121
conda env config vars set -n verl VERL_ATTN_IMPLEMENTATION=sdpa VLLM_ATTENTION_BACKEND=XFORMERS TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
conda deactivate
conda activate verl
```

`requirements.freeze.txt`에는 검증 시점 전체 Python 버전을, `environment.yml`에는 conda 환경을 보관한다. 다른 장비에서 재설치 시 editable 경로를 맞춰야 한다.

## Megatron 및 학습 스크립트

`script/run_fsdp.sh`, `script/run_megatron.sh` 두 런처를 제공한다. 별도 `verl-megatron` 환경과 하드웨어 제약은 [MEGATRON.md](MEGATRON.md), 최종 검증 범위는 [SMOKE_RESULTS.md](SMOKE_RESULTS.md)를 참고한다.

실제 GSM8K GRPO 1스텝도 종료 코드 0으로 검증했다. 상세 수치와 범위는 `SMOKE_RESULTS.md`의 추가 검증 항목을 참고한다. 실행 스크립트는 백그라운드 Torch 컴파일을 비활성화하고 산출물을 `outputs/`에 저장한다.
