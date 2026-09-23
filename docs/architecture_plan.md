이 문서는 초기 설계안이다. 적용된 prototype의 현재 경로·실행법·제약과 검사 범위는 [구현된 구조](architecture.md)를 기준으로 한다. 아래의 확장 목표 전체가 구현됐다는 의미는 아니다.

이 문서는 `/home/sungmin/Parse-Focusing`의 구조를 기준으로 작성한 개편 계획이다. 2026-09-14 사용자 정정을 반영했으며, 이후 적용한 구현 내용은 위 링크에 정리했다.

이번 개편의 확정 범위는 다음과 같다.

- RL과 SFT는 독립적인 학습 pipeline이다. RL은 verl, SFT만 Lightning을 사용한다.
- RL과 SFT 실행은 `scripts/train/`, evaluation 실행은 `scripts/eval/`에서 관리한다.
- SFT는 수학 풀이 생성과 오류 분류 둘 다 초기 prototype에 포함한다.
- RL은 사전 SFT 여부와 관계없이 지정한 초기 모델을 받는다. 일반 모델, SFT 모델, base + LoRA adapter를 동일한 로딩 계약으로 지원한다.
- metric은 입력 데이터에 대한 모델의 반응을 계산한다. callback은 모델 자체의 파라미터·구조·상태를 관찰한다.
- Parse-Focusing의 디렉터리, Hydra group, 부품 간 책임 경계를 함께 모방한다.
- 기존 실험의 설정·데이터·checkpoint·평가 의미와 재현성을 보존한다.

이 계획의 구분 기준은 [참조 구현의 책임 경계와 RL 대응](architecture_boundaries.md)에 정리했다. model/modules, model/loss, task/trainer, metric/모델 관찰, 계산/표시, data/sampler 등의 실제 의존성을 확인하고 같은 기준을 RL에 적용했다. 아래 구조와 이전 단계는 이 기준을 따른다.

참조에서 가장 먼저 모방할 기준은 [callbacks/geometry.py](../../Parse-Focusing/scripts/parser/callbacks/geometry.py)에 직접 설명되어 있다. 이 모듈은 batch가 아닌 모델의 embedding과 rule tensor를 읽는다. 동일한 모델 관측값을 매 batch에서 반복 계산하는 대신 epoch 끝에 관찰하며, 계산 함수는 `metric/functional`과 공유한다.

| 구분 | 읽는 대상 | Parse-Focusing에서의 예 | math_reasoning에서의 예 |
|---|---|---|---|
| metric | 데이터, 모델 출력, 정답 | 예측 parse와 gold를 비교하는 F1 | 정답률, pass@k, 분류 accuracy/macro-F1, 데이터에 대한 loss 집계 |
| callback | 모델 자체의 파라미터·구조·상태 | grammar rule 분포, symbol embedding geometry | 모델 weight norm, LoRA 가중치 변화, embedding/adapter spectrum |
| functional | 전달받은 수치·tensor | entropy, norm, effective rank 계산 | 관찰 callback과 사후 분석에서 공유하는 순수 계산 함수 |

이 구분은 이번 프로젝트의 설계 기준이다. callback 자체가 모델에 내장된 rule이라는 뜻이 아니라, callback이 그 rule과 모델 내부 상태를 읽어 관찰한다는 뜻으로 적용한다. LLM에는 PCFG와 같은 명시적 grammar rule tensor가 없으므로 실제 대응 대상은 가중치, embedding, LoRA adapter 등의 구조다. 입력에 따라 달라지는 attention·hidden state·출력 분포는 데이터 의존 관찰이므로 모델 자체만 읽는 callback으로 포장하지 않는다.

관찰 callback은 학습 데이터, gradient, reward, optimizer를 변경하지 않는다. checkpoint 저장, validation 주기, 종료 조건은 trainer/runner의 실행 관리 책임으로 둔다. 프레임워크의 기본 checkpoint callback을 사용하는 것은 연구용 모델 관찰 callback의 책임과 구분한다.

참조와 목표 위치의 대응은 다음과 같다.

| Parse-Focusing | 목표 위치 | 모방하는 책임 |
|---|---|---|
| `main.py` | `main.py` | Hydra 조합 후 작은 진입점에서 실행 연결 |
| `config/experiment/` | 같은 위치 | 실험별 부품 선택 |
| `config/stage/` | 같은 위치 | fit/test/predict 선택 |
| `config/{model,data,task,loss,optimizer,lr_scheduler,trainer}/` | 같은 위치 | 책임별 설정 조합 |
| `config/{train_config,test_config}/` | 같은 위치 | 학습 인자와 평가 인자 분리 |
| `config/metrics/leaf/` | 같은 위치 | 데이터 의존 metric 단위 설정 |
| `config/callbacks/` | 같은 위치 | 관찰 대상과 관찰 주기를 선택하는 설정 |
| `scripts/parser/` | `scripts/reasoning/` | 도메인 공통 부품 |
| `scripts/parser/lit_model/` | `scripts/reasoning/lit_model/` | SFT task를 감싸는 LightningModule |
| `scripts/parser/metric/parsed_data.py` | `scripts/reasoning/metric/prediction_data.py` | 여러 metric이 공유하는 데이터/예측 결과 |
| `scripts/parser/callbacks/geometry.py` | `scripts/reasoning/callbacks/geometry.py` | 모델 자체를 읽는 관찰 |
| `scripts/parser/metric/functional/` | `scripts/reasoning/metric/functional/` | metric/callback/분석이 재사용하는 순수 함수 |
| `scripts/{preprocessing,postprocessing,analysis,utils}/` | 같은 위치 | 데이터 준비, 변환, 분석, 기반 기능 |

추가 근거: [참조 main.py](../../Parse-Focusing/main.py), [experiment 조합](../../Parse-Focusing/config/experiment/npcfg.yaml), [metric 조합](../../Parse-Focusing/config/metrics/model/npcfg.yaml), [metric factory](../../Parse-Focusing/scripts/utils/metric.py), [관찰 callback](../../Parse-Focusing/scripts/parser/callbacks/model_analyzer.py).

목표 디렉터리는 다음과 같다. `config/` 단수, `scripts/` 복수, 코드 `metric/` 단수와 설정 `metrics/` 복수까지 참조에 맞춘다. 각 폴더는 해당 기능을 이전하면서 실제 구현과 함께 만든다.

```text
math_reasoning/
├── main.py
├── config/
│   ├── config.yaml
│   ├── experiment/            # rl_math, sft_solve, sft_classify, eval_math, eval_classify
│   ├── stage/                 # fit, test, predict
│   ├── task/                  # rl, sft_solve, sft_classify, evaluation
│   ├── model/
│   │   └── adapters/          # 모델과 LoRA 구성
│   ├── data/
│   │   ├── modules/
│   │   ├── sources/           # MATH, PERL 등
│   │   └── prompts/
│   ├── train_config/
│   ├── test_config/           # 생성 조건 또는 분류 추론 조건
│   ├── rollout/                # 답변 생성 수와 sampling recipe
│   ├── grader/                 # 판정 구현/버전/환경
│   ├── reward/                 # 개별 reward와 가중 합
│   ├── advantage/              # GRPO group normalization 등
│   ├── loss/                   # policy/regularization/reduction 구성
│   ├── optimizer/
│   ├── lr_scheduler/
│   ├── trainer/               # verl / lightning, 저장·평가 주기
│   ├── evaluation/
│   ├── metrics/
│   │   ├── leaf/              # accuracy, pass_at_k, macro_f1, confusion_matrix
│   │   ├── model/             # 필요한 모델별 metric bundle
│   │   ├── solve.yaml
│   │   └── classify.yaml
│   ├── callbacks/
│   │   ├── training/          # 학습 중 모델 관찰 설정
│   │   ├── analysis/          # 모델 geometry, 가중치 변화 관찰
│   │   ├── train.yaml
│   │   └── analysis.yaml
│   ├── logger/
│   └── runtime/               # 경로, 실행 환경, GPU/dtype, backend 옵션
├── scripts/
│   ├── train/
│   │   ├── rl/
│   │   │   ├── runner.py
│   │   │   ├── worker.py
│   │   │   ├── task.py        # RL 부품의 계산 계약 연결
│   │   │   ├── config.py      # 프로젝트 설정 → native verl 설정
│   │   │   ├── backend.py     # native role/registry/worker 연결
│   │   │   ├── observation.py # 분산 actor의 모델 관찰 연결
│   │   │   └── checkpoint.py
│   │   └── sft/
│   │       ├── runner.py      # Lightning 조립
│   │       ├── worker.py
│   │       └── observation.py # Lightning의 모델 관찰 연결
│   ├── eval/
│   │   ├── pipeline.py
│   │   ├── inference/         # generation, classification, backend adapters
│   │   ├── graders/
│   │   ├── scope.py
│   │   └── artifacts.py
│   ├── reasoning/
│   │   ├── model/             # 공통 로딩 규약과 모델 관찰 인터페이스
│   │   ├── modules/
│   │   ├── lit_model/         # LightningSolveSFT, LightningErrorClassifier
│   │   ├── data/              # schema, data_module, dataset, collator, sampler
│   │   ├── rollout/           # 생성 인터페이스, sampling spec, trajectory
│   │   ├── reward/
│   │   ├── advantage/         # estimator 계약과 native 연결
│   │   ├── loss/              # 개별 항/가중치/reduction, native objective 연결
│   │   ├── metric/
│   │   │   ├── base.py
│   │   │   ├── prediction_data.py
│   │   │   ├── accuracy.py
│   │   │   ├── pass_at_k.py
│   │   │   ├── classification.py
│   │   │   └── functional/
│   │   ├── callbacks/
│   │   │   ├── base.py
│   │   │   ├── model_analyzer.py
│   │   │   └── geometry.py
│   │   └── helper/
│   ├── preprocessing/
│   ├── postprocessing/
│   ├── analysis/              # 공통 io와 runs, inspect, compare, plot, report
│   └── utils/                 # config, logger, seed, manifest, atomic I/O
├── data/                      # 현재 dataset/의 최종 대응 위치
├── ext/
├── requirements/              # RL/SFT/eval 환경 재구축 기록
├── tests/
├── docs/
└── README.md
```

코드 구조와 물리 데이터 저장 경로는 구분한다. 대용량 데이터와 기존 `/data/...` run은 먼저 이동하지 않는다. 기존 실행 경로와 분석 reader의 호환성을 확인한 뒤 `dataset/`의 최종 배치를 정리한다.

실행과 의존 방향은 다음과 같다.

- `main.py`는 설정을 조합하고 RL, SFT, eval 중 하나를 선택한다.
- RL task는 model/rollout/grader/reward/advantage/loss의 계약을 연결한다. runner는 run과 프로세스를 관리하고 backend가 verl worker의 실행에 연결한다. RL에서 Lightning을 요구하지 않는다.
- SFT runner는 동일한 model/data/metric/callback 구성 방식으로 풀이 생성 task 또는 오류 분류 task를 Lightning에 연결한다.
- eval은 지정한 모델로 추론하고 task별 metric을 계산한다. 생성 평가와 분류 평가 모두 지원한다. 생성 인터페이스는 reasoning/rollout과 공유하고, backend별 생성 구현을 eval과 RL 양쪽에 복제하지 않는다.
- train/eval은 `reasoning` 공통 부품을 사용한다. eval에서 train 실행기나 Lightning task를 import하지 않는다.
- 모델 관찰 callback은 학습 중 또는 독립 eval의 모델 로딩 후에도 같은 계산 구현을 사용할 수 있다. 저장된 예측만 재집계할 때는 모델 관찰을 실행하지 않는다.
- 모듈 import만으로 CUDA 초기화, 환경변수 변경, 데이터 생성이나 학습이 시작되지 않아야 한다.

개편 후 실행 예시는 다음과 같다. `model.initialization.path`는 모델 artifact의 경로 또는 revision을 받는 목표 인터페이스다. adapter artifact이면 manifest에서 정확한 base model과 tokenizer를 찾도록 한다.

```bash
python main.py experiment=rl_math model.initialization.path=/models/base
python main.py experiment=rl_math model.initialization.path=/models/sft_export
python main.py experiment=sft_solve
python main.py experiment=sft_classify
python main.py experiment=eval_math stage=test model.initialization.path=/models/export
python main.py experiment=eval_classify stage=test model.initialization.path=/models/export
python main.py experiment=rl_math callbacks=analysis
python main.py experiment=eval_math metrics=solve
python main.py experiment=rl_math --cfg job --resolve
python main.py -m experiment=rl_math seed=17,29
```

RL은 초기 artifact의 학습 이력으로 실행 경로를 나누지 않는다. 로더가 검증할 것은 모델 구조, base revision, tokenizer, adapter의 rank/alpha/target module 및 저장 형식이다. full model, merged SFT model, base + adapter를 공통 ModelSpec으로 정규화하고 backend loader에 전달한다. 학습 이력은 출처 metadata로 기록한다. LoRA로 초기화할 때 merge 또는 재부착으로 유효 가중치가 이중 적용되지 않는지도 확인한다.

초기 가중치 로딩과 실행 재개는 별도 동작이다. 새 RL run은 지정한 모델에서 새 optimizer로 시작하고, RL resume는 native checkpoint의 optimizer/scheduler/RNG 등을 복원한다. 별도의 SFT → RL 자동 실행기나 pipeline 연결 설정을 이번 구조의 전제로 두지 않는다. 현재 RL의 알고리즘 설정은 이전 과정에서 보존하며, KL reference 정책 변경을 구조 개편의 사용자 결정 사항으로 요구하지 않는다.

Hydra experiment는 참조처럼 부품만 조합한다. 아래는 목표 예시다.

```yaml
# config/experiment/rl_math.yaml
# @package _global_
defaults:
  - /data/modules@data: math
  - /data/sources@data: math500_frozen
  - /data/prompts@data.prompt: reasoning_xml
  - /task: rl
  - /model: qwen25_math_1_5b
  - /train_config: grpo
  - /rollout: sample8
  - /test_config: greedy
  - /grader: limit_rlvr
  - /reward: math
  - /advantage: grpo
  - /loss: grpo
  - /optimizer: adamw
  - /lr_scheduler: constant
  - /trainer: verl
  - /evaluation: math_validation
  - /metrics: solve
  - /callbacks: train
  - /logger: wandb_csv
  - /runtime: local_verl
  - _self_
seed: 17
```

metric leaf를 `task.val_metrics`, `task.test_metrics`에 구성하는 방식과 callback dictionary를 조립하는 방식도 참조에서 가져온다. task가 요구하는 입력과 모델이 제공하는 관찰 대상을 설정 조합 시 검사한다. 분류 출력에 pass@k를 붙이거나 존재하지 않는 LoRA를 관찰하도록 설정하면 실행 전에 설명 가능한 오류를 낸다.

현재 `configs/verl_math.yaml`은 native verl 설정을 root에 합친다. 새 공통 `model/data/trainer`와 이름이 충돌하므로 프로젝트 설정과 native config를 분리한다. `scripts/train/rl/config.py`가 변환을 담당하고, 사용자 config와 실제 native config를 모두 저장한다. 같은 값을 두 곳에서 설정하지 않도록 mapping과 override 규칙을 명시한다. backend가 지원하지 않는 설정을 조용히 무시하지 않는다. GRPO advantage, policy objective, KL/entropy와 optimizer 실행은 각자의 설정 계약을 통해 기존 native 구현에 연결한다. 각 extension point와 필드 mapping은 책임 경계 문서에 기록한다.

Data sampler는 어느 문제를 배치에 넣을지 정하고 rollout은 그 문제에서 어떤 답변을 몇 개 생성할지 정한다. reward는 답변별 학습 점수를 만들고 advantage는 같은 문제 그룹 안에서 그 점수를 변환한다. policy loss는 현재/이전 log-prob와 advantage를 받아 학습 목적함수를 계산한다. 이 부품들은 각각 입력·상태·교체 이유가 다르므로 하나의 RL script나 단일 loss 설정에 섞지 않는다.

metric/callback을 실제로 조립하는 계약은 다음과 같다.

| 부품 | 계약 | 초기 prototype |
|---|---|---|
| Data module | sample ID, split, task별 입력/target을 제공 | MATH 풀이 생성, PERL 오류 분류, RL/eval prompt view |
| Rollout | prompt + policy + sampling spec → trajectory | 현재 verl 생성 연결과 평가용 생성 인터페이스 |
| Grader / reward | 판정 서비스 / 개별 학습 점수의 composition | 기존 수학 판정과 correctness/format/numeric 분리 |
| Advantage | reward + prompt group + mask → advantage | native GRPO normalization 연결 |
| Loss | model output + 학습 target → 미분 가능한 항별/전체 objective | SFT token loss, RL policy/KL/entropy 및 reduction |
| Metric | `update(prediction_data) / compute() / reset()` | 생성 correctness/pass@k/길이, 분류 accuracy/macro-F1/confusion matrix |
| Model observation | 요청한 파라미터·구조를 snapshot 또는 통계로 제공 | weight norm, LoRA의 유효 가중치 변화 관찰 |
| Callback | 주기에 맞춰 모델을 읽고 관찰 결과를 기록 | weight/adapter 관찰, 선택적으로 spectrum/effective rank |
| Functional | tensor/숫자를 받아 계산 결과 반환 | norm, spectrum 요약, confusion 집계, pass@k |
| Logger/writer | metric과 callback의 결과 및 원본 저장 | JSONL/CSV, 선택적 W&B |

계산된 metric이 scalar인지 curve/matrix인지는 metric과 callback을 나누는 기준이 아니다. confusion matrix나 데이터 의존 곡선은 metric이 계산하며, 시각화/저장 adapter는 그 결과를 읽는다. 참조의 `*_vis_metrics`처럼 계산·cache·reset을 한 주체가 소유하고 callback마다 같은 metric을 다시 계산하지 않는다.

`PredictionData`는 공통 식별자와 task별 결과를 표현한다. 생성 결과는 prompt/completion/answer/judgement를, 분류 결과는 example ID/예측 label/정답 label/필요한 scores를 가진다. 분류 결과를 임의의 생성 문자열로 바꾸지 않는다. metric마다 필요한 필드를 선언하고 부족하면 오류를 낸다. loss의 token 집계와 example 단위 accuracy의 분모도 구분한다.

정답·SFT target은 추론 prompt와 분리한다. Grader는 생성 후 completion과 gold answer를 받는다. 동일한 수학 판정 구현을 RL reward와 eval에 주입하되, correctness metric은 reward weight와 무관하게 유지한다. pass@k는 문제·모델·생성 조건별 실제 표본으로 계산하고 k > n을 거부한다. 분산 sampler padding과 재개로 인한 중복은 식별자로 제거한다. train/val/test metric은 서로 독립된 상태를 가진다.

모델 관찰은 참조처럼 batch 수와 독립적인 주기로 실행한다. 예를 들어 embedding geometry는 epoch/step 주기마다 한 번 계산하며, 같은 모델 snapshot의 값을 매 validation batch마다 계산해 평균하지 않는다. 원본 tensor/statistics와 step/model ID를 저장하면 사후 분석에서 같은 순수 함수를 적용할 수 있다.

LLM에서의 초기 관찰 대상은 가중치 norm과 adapter의 유효 변화량이다. LoRA A/B 각각의 norm과 실제 가중치 변화의 norm은 구분해서 기록한다. spectrum/effective rank는 대상 layer와 주기를 설정으로 제한한다. gradient 관찰을 추가하는 경우에도 이미 계산된 상태를 읽는 동작이며 gradient 변경 기능으로 확장하지 않는다. 입력이 필요한 진단은 데이터 의존 metric 경로로 명시한다.

RL의 모델은 분산 worker에 있으므로 callback이 root process의 로컬 model 객체를 읽는 방식으로 구현할 수 없다. 관찰 adapter가 같은 policy step의 파라미터나 충분 통계를 수집해 공통 관찰 계산에 연결해야 한다. shard별 norm은 제곱합으로 환산할 수 있지만 shard별 spectrum을 평균해서 전체 spectrum으로 보고하면 안 된다. 지원하지 않는 관찰은 명시적으로 거부하며, 관찰을 위해 GPU에 별도의 전체 모델을 올리지 않는다. SFT는 Lightning hook에서 같은 관찰 구현을 호출한다.

관찰 on/off가 모델 가중치·optimizer 상태·학습 RNG를 바꾸지 않아야 한다. 추론 mode나 cache 설정을 잠시 바꿨다면 복원하고, 관찰용 sampling이 있으면 독립 RNG를 사용한다. 관찰 결과에는 run/session/step/config ID를 넣어 재개 때 중복 기록을 식별한다. 시간 제한 실험에서는 관찰 비용이 완료 update 수에 영향을 줄 수 있으므로 시간과 호출 횟수도 기록한다.

풀이 생성 SFT와 오류 분류 SFT는 둘 다 초기 prototype의 완료 조건이다.

| task | 학습 | 평가 |
|---|---|---|
| 풀이 생성 | prompt/padding을 제외한 target token loss | 생성 후 수학 정답률, 길이, 필요한 pass@k |
| 오류 분류 | 기존 label/code-token 방식과 mapping을 보존한 supervised loss | accuracy, macro-F1, confusion matrix |

분류 prototype은 기존 binary/multi 및 single-error 설정을 recipe로 표현한다. token label이 여러 token으로 나뉘는 조합은 기존 전제를 검사해 거부하거나 별도의 명시적 scoring 구현이 있어야 한다. 모델을 바꿔도 A/B/C/D가 항상 한 token이라고 가정하지 않는다. 생성 task와 분류 task는 model/optimizer/관찰 callback을 공유하고 dataset view, target 구성, metric을 각각 선택한다.

재현성과 기존 동작 보존 조건은 다음과 같다.

1. 사용자/native config, CLI override, config schema, source/submodule revision과 실제 로컬 변경, Python 환경을 기록한다. 현재 root와 `ext/verl`의 미커밋 변경도 재현 자료에 반영해야 한다. 참조처럼 model/data seed를 구분하고 rollout/eval/관찰 seed도 각 실행 상태에 연결한다.
2. 데이터 source hash, split/sample/family ID, model/tokenizer revision, adapter metadata, prompt 원문과 최종 tokenization 조건을 동결한다. eval에는 train/validation/test 겹침과 전체 MATH-500 평가 범위를 기록한다.
3. resume는 저장된 config와 native checkpoint를 사용한다. optimizer/scheduler/RNG/sampler/scaler 상태와 callback 기록 위치를 검증한다.
4. 현재 [FSDP checkpoint manager](../ext/verl/verl/utils/checkpoint/fsdp_checkpoint_manager.py)의 확인한 extra state에는 scheduler와 RNG 저장이 보인다. AMP scaler와 비동기 rollout 재개 상태는 추가 검증 대상으로 두며 bitwise 동일 재개를 미리 보장하지 않는다.
5. checkpoint는 모든 rank의 저장이 끝난 것을 확인한 뒤 `last`를 갱신한다. 기존 atomic publication과 retention 검사를 유지한다. best 선택은 validation과 실제 해당 step의 checkpoint를 연결한다.
6. W&B를 꺼도 metric, 모델 관찰값, 생성·채점 원본과 config를 로컬에 저장한다. 예전 run은 다시 쓰지 않고 호환 reader로 읽는다.
7. RL validation은 기존 rollout worker를 재사용한다. 독립 eval은 같은 입력 계약을 사용하되 Transformers와 vLLM이 동일한 seed에서 같은 출력을 만든다고 가정하지 않는다.

현재 코드의 이전 위치는 다음과 같다.

| 현재 파일 | 이전 단위 |
|---|---|
| `script/train_verl.py` | RL runner/worker/task/config/backend/checkpoint; 판정·reward component/composition, 공통 data/logger |
| `ext/verl`의 기존 RL 계산 | 파일 복제 없이 rollout/advantage/policy loss/regularization의 native 확장 지점·mapping에 연결 |
| `script/stages/train_supervised_stage.py` | 풀이·분류 Lightning task, dataset/collator/loss, SFT worker |
| `script/stages/run_supervised_stage.py` | 독립 SFT runner와 task별 experiment 설정 |
| `script/overfit_four.py` | 생성·평가를 eval에 분리; SFT loop는 Lightning prototype으로 이전하며 동작 차이 검증 |
| `script/eval_four_grpo.py` | eval pipeline/inference/artifacts |
| `script/limit_rlvr_grade.py` | eval/graders와 별도 grader worker 환경 |
| `script/analysis/math_metrics.py` | metric/functional |
| `script/run_timed_grpo.py` | RL runner의 실행 관리와 명시적인 실험 설정 |
| `script/run_math500_continuation.py`, `training_handoff.py` | 기존 실행과 결과를 위한 호환 경로로 보존; 새 공통 구조의 의존성에서 제외 |
| `script/math500_eval_scope.py` | eval/scope와 명시적 evaluation config |
| `script/analysis/prepare_*` | preprocessing |
| plot/report scripts | 참조형 analysis 하위 폴더 |
| `configs/*.yaml` | config/experiment와 개별 config group |

기존 Lightning SFT의 import 시 rank/GPU 조작, tokenizer/model 준비, trainer 실행은 worker entry와 명시적 생성 함수로 옮긴다. 학습률, token masking, accumulation, precision을 파일 이동과 함께 임의 변경하지 않는다. 직접 torch loop를 Lightning으로 옮기는 경우에는 별도 동작 비교가 필요하다.

진행 순서와 완료 조건은 다음과 같다.

| 순서 | 작업 | 완료 조건 |
|---|---|---|
| 0 | 현재 동작 기준 동결 | 설정·데이터·source 상태·고정 예측/채점·checkpoint fixture와 기존 검사 결과 보존 |
| 1 | 참조형 구조와 Hydra 조합 | config/scripts 배치, CPU compose, import 무부작용, 기존 CLI 호환 |
| 2 | data/결과 표현/grader/metric과 모델 관찰 분리 | 고정 입력의 metric 일치, batch 없이 모델 관찰 가능, 순수 계산과 표시 분리 |
| 3 | RL 계산 부품 분리와 독립 실행 연결 | rollout/reward/advantage/loss 계약과 native mapping, 초기 모델 종류와 무관한 loader, native 계산 및 재개 의미 보존 |
| 4 | 두 SFT prototype | 풀이 생성과 오류 분류 모두 model/data/metric/callback을 설정으로 조립 |
| 5 | 공통 eval과 관찰 연결 검증 | 각 학습 결과를 독립적으로 평가, 생성·분류 지원, 동일 callback 계산의 backend 연결 |
| 6 | 분석·환경·문서와 호환 경로 정리 | 원본 기록으로 재집계, 환경 재구축 자료, 옛 run reader, README 정리 |

검증은 구조가 제공한다고 주장하는 기능에 맞춰 진행한다. 계획 수정 중 GPU 학습이나 실행 테스트는 수행하지 않는다.

- CPU: config compose/target resolve/instantiate/native mapping/오타, data split/hash, prompt/tokenization, 풀이·분류 target masking, grader/reward fixture, advantage group/mask, loss 항별 값·gradient/reduction, metric 분모/중복 집계, 모델 artifact 호환성, callback 순수 계산, checkpoint publication/retention.
- 기존 회귀: launcher/grader/metrics/timed GRPO/eval scope/continuation 검사를 유지하고 새 모듈과 호환 경로에 연결한다.
- 소규모 GPU: RL을 서로 다른 초기 모델로 실행, 풀이 SFT와 분류 SFT 각각 실행, export reload 및 독립 eval, native checkpoint resume.
- 관찰 경계: metric 변경이 모델 관찰값을 바꾸지 않는지, 모델 관찰 callback이 데이터 loader 없이 실행되는지, 표시 on/off가 metric compute/reset 횟수를 바꾸지 않는지, 모델 관찰 on/off가 파라미터·optimizer·RNG 상태를 바꾸지 않는지 검사한다.
- backend 관찰: 분산 shard의 norm 집계, 같은 step의 snapshot, 지원 불가능한 spectrum 요청 처리, 기록된 가중치와 관찰값 일치 확인.
- resource/import: RL/eval이 Lightning을 요구하지 않는지, CPU compose가 CUDA를 초기화하지 않는지, 관찰과 validation이 중복 전체 모델을 올리지 않는지 확인한다.

완료 기준은 Parse-Focusing처럼 실험 recipe로 model/data/loss/metric/callback을 조합할 수 있고, 데이터 의존 계산과 모델 자체 관찰이 코드에서도 분리되어 있는 것이다. 독립 RL, 두 종류의 SFT, 독립 eval이 이 구조를 공유해야 한다. 앞서 제시했던 세 질문은 이번 범위의 미결정 사항으로 남기지 않는다.
