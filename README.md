# Math_reasoning_rl

`../Parse-Focusing`의 책임 구분과 디렉터리 구성을 따라 RL, SFT, evaluation을 독립적으로 조립한다. 새 설정은 `config/`, 실제 코드는 `scripts/`에 있다. **Lightning은 SFT에만 사용하고 RL은 기존 native verl을 사용한다.**

- `scripts/train/rl/`: native 설정 변환, 실행, 데이터 동결, checkpoint와 모델 관찰 연결.
- `scripts/train/sft/`: 풀이 생성과 오류 분류의 독립 Lightning pipeline.
- `scripts/eval/`: 모델 추론, 저장된 예측 재채점, 공통 metric 집계.
- `scripts/reasoning/`: model/modules, data/sampler, rollout, reward, advantage, loss, metric, 모델 관찰 callback.
- `config/experiment/`: 위 부품을 선택하는 RL·두 SFT·두 evaluation recipe.

`metric`은 입력과 모델의 예측을 받고, 연구용 `callback`은 모델 자체의 가중치·LoRA 구조를 관찰한다. 순수 계산은 `metric/functional/`에서 공유한다. SFT를 거친 모델과 base 모델 모두 같은 RL 초기화 인터페이스를 사용한다.

구조, 실행 예시, 재현성 규칙, backend 지원 범위는 [구현된 구조와 사용법](docs/architecture.md)에 정리했다. 참조 코드에서 배운 구분 기준은 [책임 경계](docs/architecture_boundaries.md), 초기 설계안은 [구조 계획](docs/architecture_plan.md)에 있다.

```bash
# 설정만 확인한다. GPU 학습을 실행하지 않는다.
/data/sungmin/math_reasoning/envs/verl-current/bin/python main.py \
  experiment=rl_math --cfg job --resolve
```

학습 entry는 `experiment=rl_math`, `experiment=sft_solve`, `experiment=sft_classify`다. 독립 평가는 `experiment=eval_math`와 `experiment=eval_classify`를 사용한다. 실제 학습·모델 추론 명령은 GPU를 사용할 수 있으므로 [실행 예시](docs/architecture.md)를 확인한다.

이전 실험의 `configs/`와 `script/` 명령은 호환 경로로 남아 있다. `python main.py --config-name verl_math`, `bash script/run_math_verl.sh ...`의 기존 사용법은 [기존 실행기 설명](script/README.md)을 참고한다. `data/`는 기존 `dataset/`에 대한 링크다. 로컬 실험 명세인 `experiments/math_error_transfer/`와 `/data`의 모델·입력·checkpoint는 Git clone에 포함되지 않는다.

## 다른 컴퓨터에서 소스 받기

```bash
git clone --recurse-submodules git@github.com:munsungmin/Math_reasoning_rl.git
cd Math_reasoning_rl
git submodule status --recursive
```

이미 clone했다면 `git pull --ff-only` 후 `git submodule sync --recursive`와
`git submodule update --init --recursive`를 실행한다. 로컬 수정이 있다면 먼저 각 저장소에 commit한다.

부모 저장소는 접근 권한이 필요하다. submodule URL은 `.gitmodules`에 기록되며,
기본 clone/update는 부모가 지정한 **정확한 commit**을 받는다.

| 경로 | 소스를 받는 저장소 | 작업 브랜치 | 공식 upstream |
|---|---|---|---|
| `ext/verl` | `munsungmin/verl` | `math-reasoning` | `verl-project/verl` |
| `ext/verl-cu121` | `munsungmin/verl` | `math-reasoning-cu121` | `verl-project/verl` |
| `ext/apex` | `munsungmin/apex` | `math-reasoning` | `NVIDIA/apex` |
| `ext/limit-of-RLVR` | `LeapLabTHU/limit-of-RLVR` | 공식 commit 고정 | `LeapLabTHU/limit-of-RLVR` |

`verl-cu121`은 같은 fork의 별도 호환성 브랜치다. Apex는 누락됐던 `apex/` Python
패키지를 기존 기반 commit의 내용으로 복원했다. 중첩 dependency도 recursive clone으로 받는다.

Conda 환경, 모델, checkpoint, `/data`의 고정 실험 입력은
Git clone과 별도로 준비해야 한다. 현재 실행 코드에는 이 서버의 `/home/sungmin` 및
`/data/sungmin` 경로가 있으므로 다른 서버에서 학습하려면 경로와 환경도 맞춰야 한다.

## 수정 사항 push와 원본 업데이트

Submodule은 계속 독립 저장소로 사용한다. `origin`은 본인 fork이고,
`upstream`은 공식 저장소다. 새 clone에는 로컬 `upstream` 설정이 전달되지 않으므로
다음 명령으로 추가한다. 현재 서버에는 이미 설정되어 있다.

```bash
git -C ext/verl remote add upstream https://github.com/verl-project/verl.git
git -C ext/verl-cu121 remote add upstream https://github.com/verl-project/verl.git
git -C ext/apex remote add upstream https://github.com/NVIDIA/apex.git
```

새 clone의 submodule은 commit에 직접 체크아웃된 상태다. 수정할 저장소에서 작업
브랜치로 먼저 전환한다. 아래는 `verl` 예시다. Apex는 `math-reasoning`,
cu121은 `math-reasoning-cu121` 브랜치를 사용한다.

```bash
cd ext/verl
git switch --track origin/math-reasoning  # 새 clone에서 한 번; 이미 있으면 git switch math-reasoning
git remote set-url --push origin git@github.com:munsungmin/verl.git

# 원본 업데이트를 가져와 내 작업 브랜치에 반영할 때
git fetch upstream
git merge upstream/main

# 내 소스를 수정한 뒤 해당 파일만 선택해서 commit하고 push
git add <수정한_파일>
git commit -m "Update experiment implementation"
git push -u origin math-reasoning
```

Apex의 공식 기본 브랜치는 `upstream/master`다. cu121은 오래된 CUDA·PyTorch
호환성 브랜치이므로 최신 `upstream/main`을 일괄 merge하지 말고 필요한 호환 수정만
검토하여 `git cherry-pick <commit>`으로 반영한다. 충돌은 내용을 확인해 해결한다.

Submodule을 먼저 push한 뒤, 부모에서 새 commit 포인터를 기록한다.

```bash
cd ../..
git add .gitmodules ext/verl ext/verl-cu121 ext/apex ext/limit-of-RLVR
git commit -m "Update dependency revisions"
git push --recurse-submodules=check origin main
```

공식 업데이트를 fetch하는 것만으로 부모의 고정 버전은 바뀌지 않는다. 소스 반영과
검증 후 부모의 submodule 포인터까지 commit/push해야 다른 컴퓨터에도 동일하게 전달된다.
