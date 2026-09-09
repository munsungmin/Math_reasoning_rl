# PERL: text mathematical step-error classification

Source paper/code: [PARC / PERL](https://github.com/SagnikMukherjee/PARC).
Raw HF repositories: [negatives](https://huggingface.co/datasets/PARC-DATASETS/error-detection-negatives), [positives](https://huggingface.co/datasets/PARC-DATASETS/error-detection-positives), [positives_perturbed](https://huggingface.co/datasets/PARC-DATASETS/error-detection-positives_perturbed).

Downloaded release: 214 negative + 292 positive + 247 perturbed chains = **753 chains**, **514 whitespace-normalized unique questions**, **4,449 annotated steps**. Counts are measured from these pinned public files, not the paper's nominal dataset size. Source texts, premises, step labels and chain-level labels are preserved unchanged under `raw/` and `cases.parquet`.

## Classification files

| Single step label | Rows in primary classifier dataset |
|---|---:|
| Mathematical_Error | 282 |
| Logical_Inconsistency | 361 |
| Accumulation_Error | 1,019 |
| Correct | 2,730 |

- `errors_3class.parquet`: 1,662 erroneous steps with exactly one of the three main labels.
- `steps_4class.parquet`: 4,392 steps, adding Correct.
- `steps_all.parquet`: all 4,449 annotated steps, including rare and multiple annotations.
- `cases.parquet`: all 753 chains with original records in `source_record_json`.

45 steps have multiple annotation entries (including two with repeated same-type entries); these are retained in steps_all but excluded from the simple single-label files. Another 12 single-label rare-type steps are excluded. This accounts for the 57-row difference between steps_all and steps_4class. Raw type occurrence counts are in manifest.class_statistics and can exceed the number of steps.

`Planning_Error` occurs at chain level (172 annotations), and must not be presented as a step-level class. `Missing_Steps` occurs 15 times at chain level and once at step level. These original annotations remain available in cases.parquet. No labels are inferred from final-answer correctness.

## Experimental splits

| Task | train | validation | test |
|---|---:|---:|---:|
| errors_3class | 1,309 | 189 | 164 |
| steps_4class | 3,390 | 511 | 491 |

Files are named e.g. `errors_3class_train.parquet`. These are **local experimental splits**, not official benchmark splits. Deterministic question-group hashing targets 80/10/10. All steps and variants of an identical whitespace-normalized question stay together. Paraphrases are not detected. Original source benchmarks include GSM8K/MATH; do not assume independence from existing math evaluation sets.

Inputs are `problem`, `preceding_steps`, `step_text`, optionally `premises_json`. Target is `label` (or `error_types` for multilabel work). Do not pass labels, annotation_json, or source_record_json as model inputs. Premises are supplied annotations, so report whether a classifier uses them.

This supports an initial three-error-type / four-class classification experiment, but has limited independent questions and class imbalance. Report macro-F1, per-class precision/recall and confusion matrices rather than only accuracy. It is not a large standalone training corpus. No classifier training or performance evaluation has been run.

## Reproduce

```bash
conda activate verl
CUDA_VISIBLE_DEVICES='' python script/prepare_perl.py
CUDA_VISIBLE_DEVICES='' python script/check_process_datasets.py
```

Revision, raw-file checksums, processed-file counts and label statistics are recorded in `manifest.json`. Source license/readme information is preserved in each raw subset's README.md. GPU jobs and verl sources are not involved in preparation.
