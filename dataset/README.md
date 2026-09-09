# Math reasoning datasets

학습 기본값은 **GSM8K**, 모델은 **Qwen/Qwen2.5-1.5B**, GPU는 **3,4,5,6**이다.

| 폴더 | 학습 | 평가 | 원본 |
|---|---:|---:|---|
| gsm8k | 7,473 | 1,319 | [openai/gsm8k (main)](https://huggingface.co/datasets/openai/gsm8k) |
| math | 7,497 | 5,000 | [DigitalLearningGmbH/MATH-lighteval (default)](https://huggingface.co/datasets/DigitalLearningGmbH/MATH-lighteval) |
| math500 | 없음 | 500 | [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) |
| aime2024 | 없음 | 30 | [HuggingFaceH4/aime_2024](https://huggingface.co/datasets/HuggingFaceH4/aime_2024) |
| aime2025 | 없음 | 30 (I 15 + II 15) | [opencompass/AIME2025](https://huggingface.co/datasets/opencompass/AIME2025) |
| math_level3_5 | 5,583 | 3,669 | 기존 MATH에서 Level 3·4·5 추출, verl 형식 |
| ineqmath | 1,252 (확장판 5,181) | dev 100 / test 200 | [AI4Math/IneqMath](https://huggingface.co/datasets/AI4Math/IneqMath), 원본 보존 형식 |

- `ineqmath`를 제외한 각 폴더의 `train.parquet` / `test.parquet`는 verl 입력 형식이다: `data_source`, `prompt`, `ability`, `reward_model.ground_truth`, `extra_info`.
- `raw/`에는 원본 파일과 원본 README를 보관한다. `manifest.json`에는 원본 revision, 라이선스 메타데이터, 개수와 SHA256을 기록한다.
- AIME 2025는 `test_I.parquet`, `test_II.parquet`도 제공한다. `test.parquet`는 이 두 파일의 합본이므로 셋을 함께 평가하면 중복 계산된다.
- AIME 2024는 upstream split 이름이 `train`이지만 로컬에서는 평가용 `test`로 보관한다.
- MATH 원본 학습 7,500개 중 정답이 빈 2개를 제외하고, 평가 문제와 공백 정규화 후 정확히 일치하는 1개를 추가로 제외했다. 제외 내역은 `math/train.rejected.json`, `math/train.eval_overlap_removed.json`에 남겼다. 정답을 임의로 채우지 않았다.
- MATH-500의 500문항은 MATH 평가 데이터에도 포함된다. 별개 성과 합산 시 독립 평가셋으로 취급하지 말 것.
- 중복 검사는 공백 정규화 후 동일 질문을 비교한다. 의미상 유사한 변형 문제까지 제거한 것은 아니다.
- 현재 verl 버전의 기본 reward dispatcher를 사용하기 위해 MATH/AIME의 `data_source`는 `DigitalLearningGmbH/MATH-lighteval`로 통일했다. 실제 원본은 `extra_info.source_repo` 및 manifest에 보존한다. GSM8K는 `####`, MATH/AIME는 `\boxed{}` 형식으로 최종 답을 요청한다.
- 원본 metadata에서 라이선스가 명시되지 않은 경우 manifest의 license는 null이다. 각 `raw/SOURCE_README.md`를 참고할 것.

재구축:

```bash
conda activate verl
cd /home/sungmin/math_reasoning
CUDA_VISIBLE_DEVICES='' python script/prepare_datasets.py
CUDA_VISIBLE_DEVICES='' python script/prepare_math_extensions.py
CUDA_VISIBLE_DEVICES='' python script/check_datasets.py
```

기존 manifest가 있으면 기록된 원본 revision을 재사용한다.
학습 시작 스크립트: `../script/run_fsdp.sh`.

추가 데이터 상세: [MATH Level 3–5](math_level3_5/README.md), [IneqMath](ineqmath/README.md).
원본 MATH를 다시 구축하면 확장 준비 스크립트도 실행해 파생 데이터를 갱신한다.

과정 오류 분류 데이터는 별도 스키마로 관리한다: [PERL (텍스트 수학, 우선 구축)](perl/README.md), [후보 검토 및 Socratic-PRMBench](prm_math/README.md). 각 폴더의 manifest를 사용하며 기존 RL용 manifest와 구분한다.

[RFM: 증명 전체 다중 라벨 분류](rfm/README.md)도 별도 구축했다. [PERL과 RFM 비교](PERL_RFM_COMPARISON.md)를 참고할 것.
