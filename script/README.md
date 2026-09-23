# verl 전용 실행

```bash
cd /home/sungmin/math_reasoning
conda activate /data/sungmin/math_reasoning/envs/verl-current

# configs/verl_math.yaml로 새 학습 시작: 기본 300 updates, seed 17
python main.py --config-name verl_math

# YAML을 편집하지 않고 이번 실행에만 적용
python main.py launcher.seed=29 trainer.total_training_steps=300 \
  actor_rollout_ref.actor.optim.lr=5e-6

# 재개 / 별도 터미널에서 로그 / 상태 확인
python main.py launcher.command=resume
python main.py launcher.command=logs
python main.py launcher.command=status
```

실제 `@hydra.main`을 사용하는 진입점이다. `python` 직접 실행은 위 conda 환경을 사용한다.
`bash script/run_math_verl.sh launcher.seed=29`처럼 shell을 사용하면 검증된 환경의 Python을
자동 선택하므로 conda를 바꾸지 않아도 된다. 인자가 없으면 기본 YAML로 학습을 시작한다.
실행 코드는 **`main.py` → `script/train_verl.py` → 현재 `ext/verl`의 `verl.trainer.main_ppo`**이며,
`verl-cu121`이나 TRL을 호출하지 않는다. 기존 `bash script/run_math_verl.sh start/resume/logs/status`
형식도 그대로 지원한다.
설정 원본은 **[`../configs/verl_math.yaml`](../configs/verl_math.yaml)** 하나다.
이 파일에 주석과 함께 학습·생성·데이터·GPU·reward·저장·평가·W&B 옵션을 모았다.
`train_verl.py`는 upstream Hydra 기본값에 이 YAML을 병합하고 실행을 관리한다.

## 설정을 바꾸는 곳

고정 4문제의 overfitting 실험은 `configs/verl_math_overfitting.yaml`을 사용한다.
기본값은 **10,000 updates × 4문제 × G8 = 320,000개 training rollout**이다.
4문제로는 epoch당 update가 하나이므로 `total_epochs`도 목표 update에 연결한다.
공통 설정의 `trainer.total_rollout`은 실제 batch와 생성 수에서 자동 계산된다.

```bash
bash script/run_math_verl.sh --config-name verl_math_overfitting
bash script/run_math_verl.sh --config-name verl_math_overfitting launcher.command=status
# 학습 완료 후 고정 test split 평가
bash script/run_math_verl.sh --config-name verl_math_overfitting launcher.command=test
# 진행 중에도 실제로 저장된 결과만 집계. 그래프 생성에는 matplotlib 필요.
/data/sungmin/math_reasoning/envs/verl-current/bin/python \
  script/analysis/report_verl_run.py /data/.../run-...
```

집계 결과는 해당 run의 `report/`에 JSON 요약, step별 CSV, 문제별 JSONL,
학습 곡선 PNG/PDF로 저장한다. 원본 `sessions/*/rollouts/*.jsonl`을 사용하며,
목표 step·rollout·최종 checkpoint가 모두 확인돼야 학습 완료로 표시한다.

| 바꾸려는 내용 | `configs/verl_math.yaml`의 위치 |
|---|---|
| 시작·재개·검사·로그·최종 test 선택 | `launcher.command`, `launcher.run` |
| 실험 이름·seed·물리 GPU·저장 기준 경로 | `launcher.experiment`, `seed`, `gpu_ids`, `base_dir` |
| 데이터 원본·난이도·예상 개수 | `launcher.dataset` |
| prompt·답변 길이 | `data.max_prompt_length`, `max_response_length` |
| update당 문제 수·문제당 생성 수 | `data.train_batch_size`, `actor_rollout_ref.rollout.n` |
| GPU당 학습 microbatch | `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` |
| 모델·LoRA rank/alpha/대상 모듈 | `actor_rollout_ref.model` |
| 학습률·scheduler·warmup·optimizer | `actor_rollout_ref.actor.optim` |
| KL·PPO clipping·entropy | `actor_rollout_ref.actor` |
| CPU offload·연산 dtype | `actor_rollout_ref.actor.fsdp_config` |
| 생성 temperature/top-p·vLLM 메모리 | `actor_rollout_ref.rollout` |
| 정답/형식/숫자형 reward 가중치·판정기 | `reward.custom_reward_function.reward_kwargs` |
| 목표 updates·저장/validation 주기 | `trainer.total_training_steps`, `save_freq`, `test_freq` |
| validation/test 생성 방식 | `actor_rollout_ref.rollout.val_kwargs` |
| 이전 checkpoint 삭제 여부 | `launcher.keep_last_only` |
| W&B entity/mode·project | `launcher.wandb`, `trainer.project_name` |
| 진행 메시지·로그 확인 간격 | `launcher.heartbeat_seconds`, `poll_seconds` |

예를 들어 답변 한도를 바꾸려면 `data.max_response_length`만 수정한다.
`max_model_len`과 기본 `max_num_batched_tokens`는 prompt+response 합계로 자동 계산된다.
목표 updates는 optimizer scheduler와 종료 판단·최종 test 조건에, GPU 목록의 개수는
world size와 checkpoint shard 검사에 연결된다. 길이를 늘린 설정의 GPU 메모리 적합성은
별도로 확인해야 한다. `check`는 CPU 검사이므로 GPU OOM 여부를 검증하지 않는다.

이 실행기는 단일 노드 FSDP LoRA/GRPO용이며, `ppo_epochs: 1`과
`ppo_mini_batch_size: ${data.train_batch_size}`를 유지해 **1 step = 1 optimizer update**를
보장한다. accumulation은 `문제 수 × G / (GPU 수 × microbatch)`로 결정된다.
현재 GPU 허용 범위는 2~6이다. `save_freq`는 양수여야 하며, 완전한 재개를 위해
checkpoint의 model/optimizer/extra 동기 저장과 `file` logger를 유지한다.
`total_epochs`도 목표 updates를 채울 만큼 충분해야 한다.

```bash
# 적용될 Hydra 설정만 출력. 데이터/모델/학습을 실행하지 않음
python main.py --cfg job --resolve

# GPU 없이 설정·동결 데이터·수학 판정기 검사
WANDB_MODE=disabled python main.py launcher.command=check

# 두 seed를 Hydra 기본 sweeper로 순차 실행
python main.py -m launcher.seed=17,29
```

YAML 선택은 `--config-name verl_math`, 다른 디렉터리는 `--config-dir /경로 --config-name 이름`을
쓴다. GPU 목록처럼 리스트를 넘길 때는 `'launcher.gpu_ids=[2,4,5,6]'`처럼 인자를 감싸면 된다.
설정 우선순위는 **upstream `ppo_trainer` 기본값 → 선택한 YAML → 명령행 override**다.
파일에 없는 upstream 옵션도 같은 이름으로 덮어쓸 수 있다. 일반 CLI 키의 오타는 Hydra가
거부하며, `+key=value`로 새 키를 추가할 때는 해당 라이브러리가 지원하는 옵션인지 확인한다.
지원되지 않는 LoRA dropout·4bit 옵션은 YAML에 적용 가능한 항목처럼 넣지 않았다.

Hydra는 작업 디렉터리를 변경하지 않고, 단일 실행에 별도 `outputs/`나 `.hydra/`를 만들지
않는다. 기존 실행기가 `/data`에 전체 설정과 로그를 보존한다. `--cfg` 출력의 `runtime.*`
placeholder는 실행 시 고유 run 경로로 바뀐다. `-m`은 개별 실행마다 고유 run을 만들며,
sweep 설정은 `/data/.../artifacts/math_error_transfer/verl/sweeps/`에 기록한다.

새 run은 환경변수와 보간을 해석한 전체 설정을 `/data/.../run-.../config.yaml`에 저장한다.
**`resume`/`test`는 해당 실행에 저장된 설정을 사용한다.** YAML을 수정해도 진행 중이던
실험의 설정은 바뀌지 않는다. 특정 실행은 `launcher.command=resume launcher.run=/data/.../run-...`으로
선택한다. 재개/최종 test에 학습 override를 함께 주면 적용된 것으로 오해하지 않도록 거부한다.
W&B의 `WANDB_MODE=offline/disabled`는 재개 시에도 지정할 수 있다. 현재 YAML을 읽지 않는
기존 호환 명령 `bash script/run_math_verl.sh resume --run /data/.../run-...`도 유지한다.
실제 세션별 로그·trajectory 경로까지 반영한 설정은 `sessions/<session>.yaml`에 남는다.
Python 실행 환경은 shell의 `VERL_PYTHON`으로 지정하며 기본값은 위 검증된 conda prefix다.

| 항목 | verl 기본값 |
|---|---|
| 모델·데이터 | 로컬 Qwen2.5-Math-1.5B, 동결 MATH-500 level 3~5 (train 264 / validation 30 / test 37) |
| GPU | 2,4,5,6에서 FSDP와 vLLM 동시 배치, TP=1. GPU 3은 사용하지 않음 |
| 업데이트 | 300, seed 17 (`--seed 29`로 두 번째 실험) |
| 배치 | update당 문제 4개 × G8 = 32 trajectories; GPU당 microbatch2 × 누적4 |
| 길이 | prompt≤256 + completion≤768 |
| LoRA | r16 / alpha32 / dropout0, q/k/v/o projections |
| 수치·메모리 | FP16 연산, FP32 저장·reduction, gradient checkpointing, CPU parameter/optimizer offload, 4bit 없음 |
| Attention | 학습 SDPA, 생성 Triton attention, eager 실행 |
| Optimizer | torch AdamW, LR 5e-6 **constant**, warmup0, weight decay0, grad clip0.1 |
| GRPO reward·KL | 2×수학 정답 + 0.5×형식 + 0.5×숫자형, KL coefficient0.01, base 공유 reference |
| 저장·평가 | 20 updates마다 저장 및 greedy validation; 마지막 완성 저장본 하나 유지 |

**방금 성공한 native verl smoke 설정을 기준으로 한다.** 이전 TRL 실험의
paged AdamW8bit·LoRA dropout0.05·linear LR·4학습 GPU+1생성 GPU와 동일하지 않다.
300-update 실행기가 GPU에서 검증됐다는 뜻은 아니며, 1-update native 경로 통과와
새 실행기의 CPU 설정/체크포인트 관리 검사를 구분한다.

```bash
# GPU 학습을 시작하지 않고 설정·동결 데이터·판정기만 검사
WANDB_MODE=disabled bash script/run_math_verl.sh check

# 두 번째 seed: start/resume/logs/status 모두 동일한 seed 옵션 사용
bash script/run_math_verl.sh start --seed 29

# 총 300 updates 완료 후, 최종 checkpoint에서 test 37문제만 평가
bash script/run_math_verl.sh test
```

Validation은 update 20,40,…,300에서 30문제를 사용한다. Test 37문제는 `test` 명령으로만
평가하며 학습·주기적 validation에 사용하지 않는다. 모두 greedy 문제당 1답변이며,
이 내부 분할의 정답률은 전체 MATH-500 공식 점수나 pass@K가 아니다.

W&B는 기본 project `math-reasoning`, entity
`sungmin00-gwangju-institute-of-science-and-technology`에 native verl 지표를 기록한다.
`WANDB_MODE=offline`/`disabled`도 지원한다. `WANDB_PROJECT`/`WANDB_ENTITY`는 새 실행 전에
지정한다. 재개는 저장된 W&B run ID를 사용하고, 최종 test는 별도 `-test` run을 사용한다.
Train loss/reward/KL/gradient 및 validation `val-core/.../acc/mean@1`은 native W&B 지표다.
터미널과 누적 `metrics.jsonl`에는 trajectory에서 계산한 `train/accuracy`도 추가한다.

- artifacts: `/data/sungmin/math_reasoning/artifacts/math_error_transfer/verl/E1/seed-0017/run-.../`
- checkpoints: `/data/sungmin/math_reasoning/checkpoints/math_error_transfer/verl/E1/seed-0017/run-.../last`

`last`는 완성된 `global_step_20`, `global_step_40`, …을 가리키는 symlink다.
새 저장이 끝나고 모든 rank 파일을 확인한 뒤 last를 원자적으로 교체하고 이전 저장본을
삭제한다. 저장 중에는 두 저장본이 잠시 존재한다. Native model/optimizer/scheduler/RNG/
dataloader 저장을 사용하며, 현재 native 구현은 AMP scaler 상태를 별도로 저장하지 않는다.

`train.log`, `metrics.jsonl`, `rollouts.jsonl`은 재개해도 누적한다. 각 세션의 원본 지표,
실제 YAML, train/validation/test trajectory는 `sessions/` 아래에 따로 보존한다.
`--run /data/.../run-...`으로 특정 verl 실행을 선택할 수 있다. 기존 TRL checkpoint는
불러오지 않으며, 사용 중인 GPU가 있으면 실행을 거부한다. 학습 실행과 Git push는 별개다.

## 실행에 필요한 파일

| 파일 | 역할 |
|---|---|
| `../main.py` | 프로젝트 루트의 실행 진입점 |
| `run_math_verl.sh` | 검증된 Python 자동 선택, main.py에 인자 전달 |
| `../configs/verl_math.yaml` | 사용자가 수정할 학습·생성·GPU·데이터·저장·평가 설정 |
| `train_verl.py` | YAML 로딩, 고정 데이터 검사, reward 연결, 최신 checkpoint 교체 |
| `limit_rlvr_grade.py` | `ext/limit-of-RLVR` 기반 수학 정답 판정 |

`stages/run_supervised_stage.py`, `stages/train_supervised_stage.py`는 Lightning 기반
SFT·오류 분류 학습 코드다. `analysis/`에는 저장된 결과의 지표 계산·그래프·판정 입력
내보내기 코드가 남아 있다. 기존 결과를 읽는 분석 경로와 새 verl run의 경로는 다르므로
해당 코드의 입력 경로를 확인한다. TRL 전용 학습/평가 실행기와 실험 큐는 제거했다.
