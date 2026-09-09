# Qwen2.5-1.5B GRPO 실행

두 학습 런처 모두 GPU `3,4,5,6`, `dataset/gsm8k/train.parquet`, `dataset/gsm8k/test.parquet`가 기본값이다.

```bash
cd /home/sungmin/math_reasoning
conda activate verl
bash script/run_fsdp.sh

conda activate verl-megatron
bash script/run_megatron.sh
```

| 런처 | 환경 | 기본 병렬화 | 현재 RTX 2080 Ti |
|---|---|---|---|
| `run_fsdp.sh` | `verl` | FSDP 4 GPU, rollout TP=1 | FP32 학습 / FP16 생성, SDPA / xFormers |
| `run_megatron.sh` | `verl-megatron` | TP=2 × PP=2, rollout TP=2 | 현재 verl 코드가 BF16 + FlashAttention 2를 요구하므로 사전 검사에서 중단 |

Megatron 런처는 지원 GPU를 위한 설정이며, 현재 GPU에서 학습 성공을 검증한 런처가 아니다. Qwen 1.5B의 KV head 수가 2이므로 TP=4 대신 TP=2 × PP=2를 사용한다. 기존 verl 소스는 이 런처를 위해 추가 수정하지 않았다.

설정만 확인 (학습하지 않음):

```bash
conda activate verl
bash script/run_fsdp.sh --cfg job
conda activate verl-megatron
bash script/run_megatron.sh --cfg job
```

추가 Hydra 인자는 마지막에 붙인다. 예: `bash script/run_fsdp.sh trainer.total_training_steps=1`.
`TRAIN_FILE`, `VAL_FILE`, `MODEL_PATH` 환경변수로 데이터와 모델을 바꿀 수 있다. 평가용 MATH-500/AIME 파일을 학습 파일로 지정하지 말 것.

두 런처는 짧은 context와 작은 batch로 시작하며, 기본 저장/정기평가는 꺼져 있다(`save_freq=-1`, `test_freq=-1`). 장시간 학습에서는 원하는 체크포인트 저장 주기와 평가 주기를 인자로 지정한다.

데이터 재구축: `python script/prepare_datasets.py`.
데이터 검증: `python script/check_datasets.py`.
GPU/라이브러리 smoke test: `bash setup-verl/run_smoke.sh` (`verl` 환경).

FSDP의 실제 GRPO 1스텝 실행을 검증했다(종료 코드 0). 초기화 제외 38.1초, 해당 batch의 정답 보상은 0이었다. 재현 로그는 `setup-verl/grpo-one-step.log`. Torch 컴파일은 환경변수로 비활성화하며, Hydra 실행 기록과 checkpoint 경로는 소스 밖 `outputs/`를 사용한다.
