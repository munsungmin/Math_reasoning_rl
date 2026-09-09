# verl-megatron 환경

```bash
conda activate verl-megatron
```

기존 `verl` 환경을 복제하고 Megatron 의존성을 별도 설치했다. 원본 `verl/` 및 기존 호환 worktree의 Megatron 소스는 수정하지 않았다.

- Python 3.10 / PyTorch 2.4.0+cu121 / vLLM 0.6.3
- Megatron-Core 0.11.0, nvidia-modelopt 0.19.0
- Apex: NVIDIA/apex `24.04`, commit `b496d85fb88a801d8e680872a12822de310951fd`, CUDA extensions 빌드
- FlashAttention 2.6.3: 공식 cu123/torch2.4/CPython3.10/CXX11_ABI=false wheel
- CUDA_VISIBLE_DEVICES=3,4,5,6, NCCL_P2P_DISABLE=1, NCCL_IB_DISABLE=1

## 현재 GPU의 제약

현재 verl v0.3.0.post1의 `ActorRolloutRefWorker.init_model()`은 `self.param_dtype = torch.bfloat16`을 고정하고, Megatron optimizer/config 역시 BF16을 설정한다. Qwen 모델 registry는 remove-padding 모델을 선택하여 FlashAttention 2 커널을 실행한다. 따라서 환경 설치와 Megatron-Core 단위 실행이 성공해도 **현재 RTX 2080 Ti에서 이 verl Megatron 학습 경로를 실행할 수 있다는 뜻은 아니다.**

RTX 2080 Ti는 compute capability 7.5다. 설치한 공식 FlashAttention 2 커널은 Ampere 이상을 요구한다. `script/run_megatron.sh`는 이 조건을 먼저 검사하고 설명과 함께 중단한다. 현재 GPU에서는 `script/run_fsdp.sh`를 사용한다.

이 이전 verl 경로의 직접 의존성은 Apex/FlashAttention이며, 최신 Transformer Engine을 추가 설치하지 않았다. GPU를 바꾸어 사용할 때도 전체 학습 및 CUDA/드라이버 호환성을 다시 검증해야 한다. 현재 Apex wheel은 SM75용이므로 Ampere 등 다른 세대에서는 해당 아키텍처로 재빌드한다.

공식 자료:
- https://verl.readthedocs.io/en/v0.3.x/start/install.html
- https://github.com/NVIDIA/apex/tree/24.04
- https://github.com/Dao-AILab/flash-attention/tree/v2.6.3

## 재현 및 검증

```bash
conda activate verl-megatron
cd /home/sungmin/math_reasoning
python setup-verl/build_apex.py
uv pip install --python "$CONDA_PREFIX/bin/python" --no-deps third_party/apex/dist/apex-*.whl
CUDA_VISIBLE_DEVICES=3 python setup-verl/smoke_megatron.py
bash script/run_megatron.sh --cfg job
```

`build_apex.py`는 빌드 과정에서만 GPU 사용 여부를 false로 처리하고 명시한 SM75용 wheel을 컴파일한다. 실제 라이브러리 런타임과 GPU 테스트에는 이 처리를 적용하지 않는다. `TORCH_CUDA_ARCH_LIST` 환경변수로 빌드 타깃을 변경할 수 있다.

최종 단위 실행 결과는 `megatron-smoke-result.json`, 로그는 `megatron-smoke.log`, 전체 패키지 버전은 `megatron-requirements.freeze.txt`에 저장한다. smoke script의 종료 코드 2는 verl 연동 또는 FlashAttention 검증 실패를 의미한다.

## 최종 실측 결과

- Megatron-Core GPU forward/backward 및 Apex FusedAdam 업데이트: PASS (TP=1)
- Apex FP16 RMSNorm forward/backward: PASS
- verl Megatron Qwen 모델 / worker import: PASS
- FlashAttention 2: FAIL — `FlashAttention only supports Ampere GPUs or newer.`
- 전체 verl Megatron RL 학습: 현재 GPU에서 미지원. 실행 성공으로 보고하지 않는다.
