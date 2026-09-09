---
license: mit
task_categories:
- text-generation
- question-answering
language:
- en
tags:
- math
- reasoning
- error-detection
- step-by-step
- parc
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
dataset_info:
  features:
  - name: data_source
    dtype: string
  - name: question
    dtype: string
  - name: ground_truth_solution
    dtype: string
  - name: ground_truth_answer
    dtype: string
  - name: model_answer
    dtype: string
  - name: steps
    sequence: string
  - name: is_correct
    dtype: bool
  - name: premise_annotation
    dtype: string
  - name: error_annotation
    dtype: string
  splits:
  - name: train
    num_bytes: 1221106
    num_examples: 214
  download_size: 436405
  dataset_size: 1221106
---

# error-detection-negatives

This dataset is part of the PARC (Premise-Annotated Reasoning Collection) and contains mathematical reasoning problems with error annotations. This dataset combines negatives samples from multiple domains.

## Dataset Description

- **Dataset Type**: negatives
- **Domains**: gsm8k, math, metamathqa, orca_math
- **Total Samples**: 214
- **Task**: Error detection in mathematical reasoning
- **Collection**: PARC

## Domain Breakdown

- **gsm8k**: 57 samples
- **math**: 44 samples
- **metamathqa**: 59 samples
- **orca_math**: 54 samples

## Features

Each example contains:
- `data_source`: The domain/source of the problem (gsm8k, math, metamathqa, orca_math)
- `question`: The mathematical problem statement
- `ground_truth_answer`: The correct answer
- `model_answer`: The model's answer (may contain errors)
- `steps`: List of individual reasoning steps
- `premise_annotation`: Detailed annotation of premises and conclusions for each step (JSON string)
- `error_annotation`: Annotations marking errors at both step and chain levels (JSON string, error descriptions removed)

Additional fields may be present depending on the source domain:
- `ground_truth_solution`: Reference solution (some domains)
- `solution`: Step-by-step solution provided by the model
- `is_correct`: Whether the model's answer is correct

## Usage

```python
from datasets import load_dataset

# Load the full dataset
dataset = load_dataset("PARC-DATASETS/error-detection-negatives")

# Filter by specific domain
gsm8k_data = dataset.filter(lambda x: x['data_source'] == 'gsm8k')
```

## Citation

If you use this dataset, please cite the original work on error detection in mathematical reasoning.
