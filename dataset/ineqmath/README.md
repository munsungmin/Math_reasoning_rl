# IneqMath / inequality

공식 원본: https://huggingface.co/datasets/AI4Math/IneqMath

`IneqMath/inequality`라는 공개 HF 저장소는 조회되지 않아 공식 배포본 `AI4Math/IneqMath`를 사용했다. 다운로드 revision과 SHA256은 `manifest.json`에 기록했다.

| 파일 | 행 수 | 용도 |
|---|---:|---|
| train.parquet | 1,252 | 학습, bound 626 / relation 626 |
| train_expanded.parquet | 5,181 | 확장 학습, bound 2,623 / relation 2,558 |
| dev.parquet | 100 | 정답 공개 개발/검증 |
| test.parquet | 200 | 정답 비공개 평가/추론 |
| raw/theorems.json | 83 | 정리 자료 |

`raw/`에는 네 split의 JSON, theorem JSON, README 및 datasheet를 그대로 보관했다. Parquet의 `source_record_json`을 JSON으로 읽으면 풀이, 정리 주석, 선택지 등 전체 원본 행을 복원할 수 있다. `problem`, `prompt`, `answer`, `task_type`, `data_id`, `source_index`, `source_repo`, `split`도 별도 열로 제공한다. 비공개 test 정답은 null이며 임의로 채우지 않았다.

이 파일들은 **원본 보존용 Parquet이며 현재 기본 verl RL 입력/reward와 바로 호환되지 않는다.** RL 연결에는 bound의 경계값과 relation의 선택지를 처리하는 변환·보상 함수가 필요하다. 정답 일치만 평가하면 증명의 타당성까지 검증한 것은 아니다. 공식 평가는 최종 답과 단계별 추론을 구분한다.

train과 train_expanded는 겹치므로 독립 학습셋처럼 합치지 않는다. 두 학습 파일 각각과 dev/test 사이에 공백 정규화 후 동일 문제는 없음을 검증했다. 의미상 유사한 문제를 탐지한 것은 아니다. 원본의 CC BY-SA 4.0 및 추가 test 사용 조건은 `raw/README.md`에 있으며, test는 학습에 사용하지 않는다.

```python
import json
import pyarrow.parquet as pq
rows = pq.read_table('dataset/ineqmath/train.parquet').to_pylist()
original = json.loads(rows[0]['source_record_json'])
print(original['solution'])
```

재구축: `CUDA_VISIBLE_DEVICES='' python script/prepare_math_extensions.py`

검증: `CUDA_VISIBLE_DEVICES='' python script/check_datasets.py`
