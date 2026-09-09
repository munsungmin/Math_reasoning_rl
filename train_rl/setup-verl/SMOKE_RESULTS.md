# 검증 결과

실행 환경: RTX 2080 Ti, NVIDIA 530.30.02, PyTorch 2.4.0+cu121.
최종 GPU 범위: **3,4,5,6**. 기존 사용자 프로세스는 변경하지 않았다.

## verl (FSDP)

- `pip check`: PASS (`pip-check.log`).
- verl/FSDP/vLLM 모듈 import: PASS (`smoke.log`).
- GPU 3,4,5,6 FP16 행렬 연산: PASS (`smoke.log`, 로컬 인덱스 0~3).
- 작은 랜덤 Qwen2 모델의 실제 verl actor forward/backward/optimizer: PASS, loss 5.52318668 (`smoke.log`).
- Ray remote worker의 verl/PyTorch import: PASS (`ray.log`).
- 실제 Qwen/Qwen2.5-1.5B FP16 vLLM 생성: PASS, "What is 2 + 2?"에 "4"로 시작하는 출력 (`qwen.log`).
- GPU 3,4,5,6 NCCL all-reduce: PASS, 모든 rank에서 sum=10 (`nccl.log`). 프로세스별 GPU 1장 노출, CUDA 초기화 직렬화, P2P/IB 비활성화 상태에서 검증했다. 이 서버에서는 여러 CUDA context 초기화가 수 분 걸렸으며, 기본 P2P 설정 시도는 완료되지 않았다.
- FSDP/Megatron 런처의 shell 문법 및 Hydra config composition: PASS (`grpo-resolved.yaml`, `megatron-resolved.yaml`).
- GSM8K, MATH, MATH-500, AIME 2024/2025 parquet schema, 정답, SHA256 및 학습/평가 정확 일치 문제 검사: PASS (`dataset-check.log`).

## 실제 GRPO 1스텝 추가 검증 — PASS

2026-09-09에 Qwen2.5-1.5B + GSM8K + GPU 3,4,5,6으로 `trainer.total_training_steps=1` 실행 후 종료 코드 0을 확인했다. FSDP/vLLM 연결, 생성, 보상 계산, log probability 계산, 역전파와 optimizer 업데이트가 완료됐다.

- 학습 batch 4, prompt당 생성 2, 최대 응답 128 tokens
- 초기화 제외 학습 step: 38.120초, actor update: 16.043초
- actor gradient norm: 0.046
- 최대 GPU allocated: 7.409 GiB, reserved: 9.699 GiB (로그 기준)
- 이번 batch의 reward/advantage는 모두 0. 생성의 87.5%가 128-token 제한에 도달했다. 순수 policy-gradient 항은 0이고 entropy regularization을 포함한 gradient가 발생했다. 학습 수렴이나 성능 향상을 검증한 결과는 아니다.
- 초기 시도의 compiler worker pool 초기화 지연을 해결하기 위해 실행 스크립트에 `TORCHINDUCTOR_COMPILE_THREADS=1`, `TORCHDYNAMO_DISABLE=1`, `PYTHONUNBUFFERED=1`을 추가했다. verl 소스 추가 변경은 없다.
- checkpoint 저장 및 별도 validation은 실행하지 않았다.

로그: `grpo-one-step.log`, 수치: `grpo-one-step-result.json`. Hydra 산출물과 향후 checkpoint는 소스 밖 `outputs/`에 보관한다.

## verl-megatron

- Conda 환경 생성, Megatron-Core 0.11.0 / Apex CUDA extension / FlashAttention 2.6.3 설치: 완료.
- `pip check`: PASS (`megatron-pip-check.log`).
- Megatron tensor-parallel linear의 GPU forward/backward + Apex FusedAdam 업데이트: PASS (TP=1).
- Apex FP16 RMSNorm forward/backward: PASS.
- verl Megatron Qwen 모델과 worker import: PASS.
- FlashAttention 2 실제 GPU 실행: **FAIL**, `RuntimeError: FlashAttention only supports Ampere GPUs or newer.`

상세 결과: `megatron-smoke-result.json`, `megatron-smoke.log`.
**현재 RTX 2080 Ti에서 이 verl 버전의 Megatron 학습은 지원되지 않는다.** 코드가 BF16 및 FlashAttention 2를 요구한다. 런처 `script/run_megatron.sh`는 이 조건을 검사해 학습 전에 중단한다. Megatron TP>1 모델 실행 및 전체 RL 학습은 검증하지 않았다.
