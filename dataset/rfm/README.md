# RFM — proof-level multi-label classification

Source: [guodadi/RFMDataset](https://github.com/guodadi/RFMDataset), revision `d73be090eb811e61d509e83e29b9b29bf5edb84f`. Source README, MIT LICENSE, problems, answers, judge outputs and judge prompt are preserved in `raw/`. This is a text mathematical proof benchmark, included separately from step-level datasets at the user's request.

## Prepared data

| File | Rows | Unit |
|---|---:|---|
| proofs.parquet | 2,354 | Nonempty proof from one model on one problem |
| ratings_all.parquet | 8,808 | Parsed judge rating of a proof; multiple judges can rate the same proof |
| classification_reference.parquet | 1,965 | One consistent rating per proof using the reference judge |

There are **200 distinct problems** (52 middle-school, 88 high-school, 60 undergraduate) and 12 answer models. The 2,400 expected answer slots contain 46 empty answers; these are recorded in manifest.missing_proofs and not silently shifted during alignment.

The comparison-ready reference judge is `gemini-2.5-pro-preview-0506`. It covers 10 answer models; the two newer models judged under other judge versions remain in proofs/ratings_all. All 200 questions are represented in the reference file. 574 reference proofs have no positive error type; 1,391 have one or more.

| Error type | Positive reference proofs |
|---|---:|
| Transformation Error | 51 |
| Over Generalization | 105 |
| Invalid Construction | 114 |
| Wrong Division | 22 |
| Circular Reasoning | 72 |
| Logic Violation | 636 |
| Hidden Assumption | 388 |
| Boundary Neglect | 101 |
| Vague Argument | 490 |
| Incomplete Proof | 769 |
| Others | 183 |

Counts overlap because one proof can have several error types. Wrong Division and other rare categories have limited support; this corpus is suitable for exploratory proof-level classification, not a large independent training set. Labels are **published LLM-as-a-judge outputs, not human ground truth**. Disagreements across judges are preserved rather than merged by an invented consensus.

## Inputs and targets

- Input: `problem` and `proof_for_judge` (the original `proof` is also preserved).
- Targets: `error_types` and an 11-element `label_vector` in manifest.categories order.
- Separate target: `overall_correctness`.
- Metadata: `problem_id`, `proof_id`, `model`, `judge`, source file and row position.
- Audit only: `judgement_text` includes the target labels and must not be fed to a classifier.

`proof_for_judge` reproduces upstream strip_thinking: when a `</think>` delimiter is present, use the final proof, falling back to the thought text if the final segment has fewer than ten words. It is a reproducible preprocessing view, not a claim that the exact historical API request is available.

No step index is invented. No train/test split has been assigned: group by problem_id if splitting, so different model proofs or judge ratings for the same question cannot leak across splits. This remains distinct from PERL's experimental splits.

## Parsing and validation

Only explicit, complete 11-category and overall labels are accepted. Unlike the upstream parser, missing labels are not defaulted to zero. 25 malformed/incomplete ratings and 167 ratings without matching proof text are excluded (see rejected.json). 15 parsed ratings have inconsistent overall/error labels; they remain flagged in ratings_all and are excluded from the reference view. Fourteen app/app_add reflection files are archived but not joined to original answers because the proof version alignment is not established.

Validated SHA256, every problem–answer–judge positional join against source arrays, labels, and unique reference proof IDs. No model inference, judge API calls or GPU jobs were run.

```bash
conda activate verl
CUDA_VISIBLE_DEVICES='' python script/prepare_rfm.py
CUDA_VISIBLE_DEVICES='' python script/check_rfm.py
```

See [PERL versus RFM](../PERL_RFM_COMPARISON.md) and [a full multi-label example](example_multilabel.json).
