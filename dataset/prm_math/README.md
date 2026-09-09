# Math/text process-error dataset audit

Selection requires textual mathematical reasoning, explicit error-category linkage to step positions, and enough support for a useful initial classification experiment. This is a measured suitability assessment, not a guarantee of classifier performance.

| Candidate | Decision | Evidence |
|---|---|---|
| PERL | Built first in `../perl/` | 753 chains; 1,662 single-label error steps in 3 main classes |
| Socratic-PRMBench | Prepared, shared-case labels | 2,915 valid cases, 20 categories, 96–158 cases/category; 8,742 typed error-step links |
| ProJudgeBench | Excluded from primary classification data | Math + pure_text gives 72 cases but only 10 unique question strings; Reasoning Error 213 steps, other types only 1–10 steps each |
| GR-Ben | Excluded | Public domains are science and general logic, without a mathematics domain |
| DeltaBench | Excluded | Section/step error location, explanation and correction, but no explicit error-type category field |
| PCBench | Excluded | conflict_type labels a flawed problem premise, not a solution step |
| RFM | Included separately at user request in ../rfm/ | 200 proof questions; structured LLM-judge labels are proof-level, not step-level |

Sources: [Socratic](https://github.com/Xiang-Li-oss/Socratic-PRMBench), [ProJudge](https://huggingface.co/datasets/julyai/ProJudgeBench), [GR-Ben](https://huggingface.co/datasets/GR-Ben/GR-Ben), [DeltaBench](https://huggingface.co/datasets/OpenStellarTeam/DeltaBench), [PCBench](https://huggingface.co/datasets/ALIENS232/PCBench).

## Socratic-PRMBench

`socratic_prmbench_math/test.parquet` contains source question, modified reasoning steps, original classification and explicit error-step indices. `test_text_only.parquet` is the identical text-only view; do not concatenate it with test.parquet. Raw JSONL files and rejected records are preserved.

**Its classification applies to the synthetic case and is linked to all listed error_steps; it is not an independently annotated error category for each step.** Unlisted steps have null labels, not invented Correct labels. Indexing is 1-based. Of 2,995 original cases, 80 have empty/invalid/out-of-range indices and are excluded; no indices were guessed or repaired. Accepted cases have 1,747 distinct source IDs; multiple variants of the same source ID should be grouped before any training/evaluation split. The 8,742 step links are not independent samples.

The local file remains an evaluation archive. No random train/test split or RL reward conversion is applied. Source math collections are mathbench-A, Omni-MATH, OlympiadBench, and gsm8k_test. Existing math evaluation corpora can overlap.

Rebuild: `CUDA_VISIBLE_DEVICES='' python script/prepare_prm_math.py`

Validate both PERL and Socratic: `CUDA_VISIBLE_DEVICES='' python script/check_process_datasets.py`

These specialized datasets use their own manifests because their schema differs from the existing verl RL parquet datasets.

RFM audit: [official repository](https://github.com/guodadi/RFMDataset), pinned evidence in `rfm_audit.json`. Detailed judge narratives sometimes refer to individual deductions, but converting those into step IDs and categories would require new extraction/annotation; it is not supplied step-level ground truth. The user subsequently authorized separate proof-level inclusion; see ../rfm/README.md and ../PERL_RFM_COMPARISON.md.
