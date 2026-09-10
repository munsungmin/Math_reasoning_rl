# Math_reasoning_rl

현재 실험은 **Qwen2.5-Math-1.5B + LoRA + TRL GRPO**이며, 학습 데이터는 **MATH-500 level 3–5**다.

```bash
cd /home/sungmin/math_reasoning
conda activate grpo-lora
bash script/run_math_main.sh resume
```

실행 방법과 실제 설정 파일은 [script/README.md](script/README.md)에 정리되어 있다. 다른 터미널에서 `bash script/run_math_main.sh logs`를 실행하면 update·loss·정답률·reward·남은 시간을 간단히 볼 수 있다.

- 현재 RL: 고정 학습 264문제, 별도 평가 67문제, seed 17/29, 실험별 300 updates.
- GPU: 학습 2·4·5·6, rollout 서버 3.
- 체크포인트: `/data/sungmin/math_reasoning/checkpoints/` 아래, 20 updates마다 최신 `last` 교체.
- 실험·판정·분석 명세: [experiments/math_error_transfer](experiments/math_error_transfer/README.md).
- `dataset/`: 이미 구축된 데이터. 여러 데이터셋이 존재하지만 현재 RL 입력은 고정 MATH-500 subset이다.
- `script/stages/`, `script/analysis/`: 후속 학습과 평가·분석 코드.
- `train_rl/`: 기존 외부 dependency checkout과 환경 기록. 현재 학습은 `grpo-lora` Conda 환경을 사용한다.

```bash
git clone --recurse-submodules https://github.com/munsungmin/Math_reasoning_rl.git
```

학습 데이터·모델·실행 artifact는 로컬 `/data` 경로에 별도로 준비되어 있다. Git clone만으로 Conda 환경이나 모델·checkpoint가 설치되지는 않는다. dependency의 기존 로컬 변경 기록은 `train_rl/patches/`에 보존되어 있다.
