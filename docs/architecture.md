현재 구현은 `../Parse-Focusing`의 `config/`, `scripts/`, `scripts/parser/` 구성을 따라간다. 도메인 패키지만 `scripts/reasoning/`으로 바꿨다. 기존 실험과 저장본을 사용하는 명령은 호환 경로로 남겼다.

```text
config/
  experiment/                   # 독립 RL, 풀이 SFT, 분류 SFT, 풀이/분류 평가
  model/adapters/               # 모델 초기화와 LoRA 부품
  data/{modules,sources,prompts}/
  task/  train_config/  test_config/  trainer/
  rollout/  grader/  reward/leaf/  advantage/
  loss/{policy,regularization,reduction}/
  metrics/leaf/                 # val/test에서 선택하는 metric 객체
  callbacks/{training,analysis}/
  optimizer/  lr_scheduler/  logger/  runtime/  backend/
scripts/
  train/
    entry.py                    # 공통 CLI, 설정 선택과 worker 실행
    rl/                         # native verl 설정·데이터·실행·저장·관찰 연결
    sft/                        # Lightning 실행·loader·관찰 hook
  eval/                         # 독립 추론, 재채점, metric 집계
    graders/  inference/
  reasoning/
    data/                       # task 입력 view, encoding, sampler
    model/  modules/            # 모델 artifact와 학습 가능한 adapter
    lit_model/                  # SFT에만 사용하는 LightningModule
    rollout/  reward/  advantage/  loss/
    metric/functional/          # 데이터 의존 통계와 순수 수치 계산
    callbacks/                  # 모델 자체 관찰
  preprocessing/  postprocessing/  analysis/  utils/
data -> dataset                 # 기존 데이터 위치를 유지하는 호환 링크
main.py
```

`metric`은 입력에 대한 모델의 반응을 받는다. accuracy, pass@k, 출력 길이, 분류 confusion matrix/F1, 조건부 entropy가 여기에 있다. 선택적 KL metric은 저장된 `log_probs`와 `reference_log_probs`를 받으며 reference 모델 선택을 대신하지 않는다.

`callbacks`의 연구용 관찰은 모델 자체를 읽는다. `WeightNorm`과 `AdapterGeometry`가 가중치와 유효 LoRA 업데이트의 spectrum을 계산한다. 추론이나 채점은 수행하지 않는다. 순수 norm/spectrum/probability 함수는 `metric/functional/`에 있고 결과 저장은 별도 writer가 맡는다. checkpoint 저장과 프로세스 관리는 trainer/runner의 책임이다.

RL은 다음 부품을 독립적으로 설정한다. 새로운 GRPO 계산을 복제하지 않고 현재 고정한 `ext/verl`의 구현을 사용한다.

| 역할 | 구현과 연결 |
|---|---|
| Rollout | `reasoning/rollout`의 sampling 설정 → native vLLM |
| Grader | `eval/graders`의 독립 Python 판정기; RL과 평가가 공유 |
| Reward | correctness/format/numeric 부품과 weights → native reward 함수 |
| Advantage | `GRPOAdvantage` → native group estimator |
| Loss | policy clipping, KL/entropy, reduction 설정 → native policy loss |
| Metric | native 결과에 stable ID·실제 token 길이를 기록하고 공통 metric에 전달 |
| 모델 관찰 | 완료된 FSDP checkpoint를 CPU에서 복원해 공통 callback에 전달 |

checkpoint의 LoRA 가중치는 저장된 dtype 그대로 읽는다. shard의 spectrum을 평균하지 않고 전체 A/B를 복원한 뒤 작은 rank 행렬의 SVD를 계산한다. 관찰용으로 추가 LM을 GPU에 올리지 않는다. 현재 이 경로는 일반 LoRA와 단일 축 FSDP mesh를 지원한다.

실행 예시는 다음과 같다. 아래 학습·모델 추론 명령은 사용자가 실행할 때 GPU를 사용한다. 이번 작업의 GPU 검사는 사용자 요청에 따라 중단했다.

```bash
# CPU에서 recipe 확인; 학습하지 않는다.
/data/sungmin/math_reasoning/envs/verl-current/bin/python main.py \
  experiment=rl_math --cfg job --resolve

# 독립 학습. SFT만 별도 Lightning 환경의 worker를 사용한다.
python main.py experiment=rl_math
python main.py experiment=sft_solve
python main.py experiment=sft_classify
python main.py experiment=sft_classify data.classification=binary 'task.labels=[A,B]'
python main.py experiment=sft_classify data.classification=single 'task.labels=[A,B]'

# 연구 부품을 교체한다.
python main.py experiment=rl_math callbacks=analysis \
  reward.weights.format=0.0 reward.weights.numeric=0.0 advantage.normalize_std=false
python main.py experiment=rl_math loss.kl_coefficient=0.0
python main.py experiment=sft_solve callbacks=none optimizer.lr=2e-5

# SFT export, RL export, base/full/merged 모델을 같은 초기화 입력으로 받는다.
python main.py experiment=rl_math model.initialization.path=/path/to/export
python main.py experiment=rl_math model.initialization.path=/path/to/base \
  model.initialization.adapter=/path/to/adapter

# 독립 평가. 출력 record를 한 번 만들고 여러 metric이 함께 읽는다.
python main.py experiment=eval_math model.initialization.path=/path/to/export
python main.py experiment=eval_math test_config=sample32 \
  'task.test_metrics.pass_at_k.ks=[1,8,32]' model.initialization.path=/path/to/export
python main.py experiment=eval_classify model.initialization.path=/path/to/export
python main.py experiment=eval_math stage=predict model.initialization.path=/path/to/export

# 저장된 예측만 재채점·집계한다. 모델/GPU가 필요 없다.
python main.py experiment=eval_math evaluation.predictions=/path/to/generated.jsonl \
  'runtime.gpu_ids=[]'

# 재개는 실행 당시 설정과 입력을 사용한다.
python main.py experiment=rl_math resume=/path/to/rl/run
python main.py experiment=sft_solve resume=/path/to/sft/run
```

adapter의 rank/alpha/target_modules는 초기화 artifact와 맞아야 한다. 기본값이 다른 adapter라면 해당 model 부품 설정을 함께 선택한다. SFT adapter에 저장된 dropout은 RL 초기화 시 run 내부 복사본에서 0으로 설정한다. 원래 가중치와 artifact는 보존하며 변환 내용을 manifest에 남긴다. 이는 현재 RL recipe의 학습 동작을 적용하기 위한 것으로, SFT를 선행해야 한다는 의미는 아니다. optimizer와 scheduler 상태를 잇는 작업은 `resume`, 모델 가중치만 가져오는 작업은 `model.initialization`으로 구분한다.

SFT는 풀이와 분류 모두 모델 forward 뒤 독립 loss를 호출한다. prompt/padding은 target loss에서 제외한다. 분류 task는 PERL의 A/B/C/D 또는 A/B 코드 계약을 검사한다. validation은 분류 metric 또는 풀이 생성/채점 metric을 사용한다. 독립 평가는 최종 checkpoint 선택이나 자동 후속 RL을 실행하지 않는다.

재현성 관련 저장·검사는 다음과 같다.

- 새 run마다 완전히 해석된 설정을 저장한다. RL은 `experiment.yaml`과 실제 native `config.yaml`을 모두 보관한다.
- 입력, 초기 모델·adapter·tokenizer의 해시, 소스 복사본, Git revision과 dirty patch, 실제 worker 환경의 패키지 목록을 남긴다.
- SFT 재개는 sampler 소비 위치, rank별 RNG, optimizer/scheduler, Lightning precision 상태를 복원한다. DataLoader prefetch는 소비 위치를 앞으로 옮기지 않는다.
- 모델 관찰과 생성 평가의 RNG를 학습 RNG와 분리한다. 관찰의 완료 상태도 checkpoint에 저장한다.
- RL은 기존 native의 dataset/RNG/checkpoint 재개를 유지한다. vLLM 비동기 scheduling까지 포함한 bitwise 재현성을 보장한다고 주장하지 않는다.
- SFT 재개 시 원본 입력·모델·소스 변경을 검사한다. RL은 동결 parquet·초기화 artifact·소스 변경을 검사한다. 환경이 달라졌는지는 저장한 `environment.json`으로 비교할 수 있다.
- 학습 작업의 성공은 worker가 정상 종료하고 최종 export가 존재할 때 기록한다. checkpoint 저장만으로 실행 성공을 표시하지 않는다.

현재 backend 제약도 설정에 드러내 두었다. RL 학습 중 validation은 `test_config=native_greedy` 또는 `native_sample32`를 쓰며 native rollout 엔진의 seed와 길이 한도를 공유한다. 독립 평가는 `greedy`/`sample32`에서 자체 seed·길이·batch를 설정한다. 현재 native launcher의 단일 노드 FSDP, GRPO, 일반 LoRA, 1 step=1 update 조건은 유지한다. 이 범위를 벗어난 알고리즘·adapter를 지원한다고 가장하지 않는다.

기존 `configs/`, `script/train_verl.py`, 분석 명령은 호환 경로로 유지한다. 분석·전처리·후처리의 실제 구현은 새 `scripts/`로 옮겼다. 과거의 특정 실험용 `script/stages/`, `overfit_four.py`, timed runner는 기존 실행 기록을 재현하기 위한 경로로 남아 있으며 새 공통 task를 구현하는 위치는 아니다. `data`는 기존 `dataset`의 링크이므로 대용량 원본을 복제하지 않았다.

검사는 다음처럼 GPU를 숨긴 환경에서 실행한다. Lightning 환경은 [requirements/sft.txt](../requirements/sft.txt)를 참고하고, RL은 현재 고정한 verl 환경을 사용한다.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 /data/sungmin/math_reasoning/envs/verl-current/bin/python \
  -m pytest tests --ignore=tests/test_math_grader.py --ignore=tests/test_composable_sft.py -q
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 /data/sungmin/math_reasoning/envs/math-supervised/bin/python \
  -m pytest tests/test_composable_sft.py -q
CUDA_VISIBLE_DEVICES='' /data/sungmin/math_reasoning/envs/limit-rlvr-grader/bin/python tests/test_math_grader.py
```

CPU 통합 검사는 실제 작은 Qwen 모델로 두 SFT task의 학습·export·독립 평가를 실행하고, 도중 중단 후 재개의 표본 순서와 최종 adapter 가중치를 연속 학습과 정확히 비교한다. config 조합, 기존 native 설정과의 알고리즘 동등성, GRPO/policy gradient, masking, metric 중복 제거, 관찰의 RNG/가중치/gradient 불변성도 검사한다. GPU RL 전체 실행의 통과 여부는 이번 완료 기준에 포함하지 않았으며 GPU 검사를 다시 시작하지 않는다.
