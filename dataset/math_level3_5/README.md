# MATH Level 3–5 (Qwen 학습용)

기존 정제 MATH의 난이도 Level 3·4·5만 추출했다. Qwen이 생성한 데이터가 아니라 Qwen 학습에 사용할 MATH 부분집합이다. 모델/토크나이저는 데이터에 고정하지 않았다.

| 레벨 | train | test |
|---|---:|---:|
| 3 | 1,592 | 1,131 |
| 4 | 1,690 | 1,214 |
| 5 | 2,301 | 1,324 |
| 합계 | 5,583 | 3,669 |

`train.parquet`, `test.parquet`는 합본이다. `train_level3.parquet`부터 `test_level5.parquet`까지 레벨별 파일도 있다. 합본과 레벨별 파일을 함께 입력하면 중복된다.

기존 train/test 분리, 정답 누락 및 평가 중복 제외를 유지했다. `extra_info.level`, `subject`, 원본 `index`를 보존했다. 원본 파일과 revision은 `../math/raw/`, `manifest.json`의 parent_files로 추적할 수 있다. MATH-500은 MATH test의 부분집합이다.

기존 FSDP 실행 스크립트에서 데이터만 선택하는 예시:

```bash
conda activate verl
cd /home/sungmin/math_reasoning
TRAIN_FILE="$PWD/dataset/math_level3_5/train.parquet" \
VAL_FILE="$PWD/dataset/math_level3_5/test.parquet" \
bash script/run_fsdp.sh trainer.total_training_steps=1
```

이 데이터로 학습을 실행한 것은 아니다. 현재 스크립트의 prompt 256 / response 128 토큰은 smoke test용이다. 긴 MATH 문제는 필터링되고 풀이가 잘릴 수 있으므로 본 학습에서는 GPU 메모리에 맞게 길이를 조정해야 한다.
