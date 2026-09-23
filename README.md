# Math_reasoning_rl

현재 실험은 **Qwen2.5-Math-1.5B + LoRA + verl GRPO**이며, 학습 데이터는 **MATH-500 level 3–5**다.

```bash
cd /home/sungmin/math_reasoning
conda activate /data/sungmin/math_reasoning/envs/verl-current
python main.py --config-name verl_math
```

학습 설정은 [configs/verl_math.yaml](configs/verl_math.yaml) 하나에서 수정한다. 옵션 설명과 실행 방법은 [script/README.md](script/README.md)에 정리되어 있다. 다른 터미널에서 `bash script/run_math_verl.sh logs`를 실행하면 누적 상세 로그를 볼 수 있다. 실행 터미널에는 update·loss·정답률·reward를 한 줄씩 표시한다.

Hydra 직접 실행은 `python main.py --config-name verl_math`이며, 검증된 `verl-current` 환경에서 실행한다. `launcher.seed=29` 같은 override, `--cfg job --resolve` 설정 출력, `-m launcher.seed=17,29` 순차 실행을 지원한다. shell에서도 같은 Hydra 인자를 전달할 수 있다.

- 현재 RL: train 264문제, validation 30문제, test 37문제, seed 17/29, 실험별 300 updates.
- W&B: train 매 update, validation 20 updates마다, test는 300 updates 완료 후 `bash script/run_math_verl.sh test`로 실행. 설정은 [script/README.md](script/README.md)에 정리되어 있다.
- GPU: 2·4·5·6에서 FSDP 학습과 vLLM rollout을 함께 실행한다.
- 체크포인트: `/data/sungmin/math_reasoning/checkpoints/` 아래, 20 updates마다 최신 `last` 교체.
- 실험·판정·분석 명세: 로컬 `experiments/math_error_transfer/`. `experiments/`는 Git에서 제외하므로 별도로 복사해야 한다.
- `dataset/`: 이미 구축된 데이터. 여러 데이터셋이 존재하지만 현재 RL 입력은 고정 MATH-500 subset이다.
- `script/stages/`, `script/analysis/`: 후속 학습과 평가·분석 코드.
- `ext/`: 외부 소스를 특정 commit으로 고정한 Git submodule. 실행기는 `/data/sungmin/math_reasoning/envs/verl-current` Conda 환경의 Python을 사용한다.

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
