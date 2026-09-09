# PERL과 RFM의 분류 단위 차이

| 항목 | PERL | RFM |
|---|---|---|
| 입력 | 문제 + 이전 step + 현재 step | 문제 + 모델 증명 전체 |
| 라벨이 붙는 단위 | 특정 step | 증명 전체 |
| 준비한 기본 과제 | 3종 오류 단일 분류 / 정상 포함 4종 분류 | 11종 오류의 다중 라벨 분류 |
| 예측 예 | step 3 = Logical_Inconsistency | 이 증명에 Over Generalization, Boundary Neglect가 있음 |
| 분류용 행 수 | 오류 1,662 / 정상 포함 4,392 | 동일 judge 기준 1,965 |
| 고유 질문 수 | 원본 전체 514 | 200 |
| 분할 | 질문 단위 실험용 train/validation/test 제공 | 아직 미분할; 문제 단위 분할 필요 |
| 제약 | 희소·복수 라벨은 원본 보존, 기본 분류셋에서 제외 | 오류 위치는 알 수 없음; judge별 판정 차이·희소 유형 존재 |

PERL에서는 한 풀이의 step 3에 Logical_Inconsistency, step 4·5에 Accumulation_Error가 각각 붙을 수 있다. step을 입력으로 주고 해당 step의 유형을 맞히는 실험이 가능하다.

RFM에서는 한 증명에 `["Over Generalization", "Boundary Neglect"]`가 함께 붙는다. **증명 안에 두 유형이 있다는 뜻이지, 어느 step이 각각 해당하는지 알려주는 라벨은 아니다.** 실제 원본 사례는 `rfm/example_multilabel.json`에 문제·풀이·judge 설명과 함께 저장했다.

단일 분류는 하나의 클래스를 선택하지만, 다중 라벨 분류는 각 유형의 존재 여부를 별도로 예측한다. RFM에서는 11개 라벨이 모두 0일 수도 있고 여러 개가 1일 수도 있다. 유형별 precision/recall/F1과 macro/micro F1을 확인하고, PERL과 RFM의 정확도를 같은 과제 점수로 직접 비교하지 않는다.

두 데이터 모두 한 질문에서 여러 step/변형/모델 풀이가 나오므로 행 수를 독립 문제 수로 해석하지 않는다. 모델 입력에 `label`, `error_types`, `judgement_text` 같은 정답 필드를 넣지 않는다.
