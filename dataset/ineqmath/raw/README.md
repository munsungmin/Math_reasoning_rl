---
language:
- en
license: cc-by-sa-4.0
size_categories:
- 1K<n<10K
task_categories:
- multiple-choice
- question-answering
- text-retrieval
- text-classification
pretty_name: IneqMath
tags:
- math
- theorem-proving
- olympiad-math
- inequality
- multi-choice
- open-ended
- fill-in-the-blank
- proofs
- natural-language-proofs
- question-answering
- arithmetic-reasoning
- algebraic-reasoning
- logical-reasoning
- math-reasoning
- multi-step-reasoning
- step-by-step-solution
configs:
- config_name: default
  data_files:
  - split: dev
    path: json/dev.json
  - split: test
    path: json/test.json
  - split: train
    path: json/train.json
  - split: train_expanded
    path: json/train_expanded.json
library_name: datasets
---

<div align="center">

  <img src="./assets/logos/ineqmath.svg" alt="IneqMath Logo" width="300"/>

  <!-- <h1 style="font-size: 40px; margin-bottom: 0;"><strong>IneqMath</strong></h1> -->

  <h2 style="font-weight: bold; margin-top: 11px;">
    Solving Inequality Proofs with Large Language Models
  </h2>

  <style>
    .custom-link {
      color: #1a0dab !important;
      text-decoration: underline !important;
      transition: color 0.2s;
      cursor: pointer;
    }
    .custom-link:hover {
      color: #d03c36 !important;
      text-decoration: underline !important;
    }
    .custom-link img {
      filter: none;
      vertical-align: middle;
    }
  </style>

  <!-- <div style="display: flex; flex-wrap: wrap; justify-content: center; align-items: center; gap: 0.7em; margin-bottom: 0.5em;">
    <a href="https://ineqmath.github.io/" class="custom-link" style="display: flex; align-items: center; gap: 0.15em;">
      🌐 <span>Project</span>
    </a>
    <span>|</span>
    <a href="https://arxiv.org/abs/2506.07927" class="custom-link" style="display: flex; align-items: center; gap: 0.15em;">
      <img src="assets/arxiv.svg" alt="arXiv" style="height:1em;"/> <span>arXiv</span>
    </a>
    <span>|</span>
    <a href="https://huggingface.co/papers/2506.07927" class="custom-link" style="display: flex; align-items: center; gap: 0.15em;">
      <img src="assets/huggingface.png" alt="hf" style="height:1em;"/> <span>HF Paper</span>
    </a>
    <span>|</span>
    <a href="https://github.com/lupantech/ineqmath" class="custom-link" style="display: flex; align-items: center; gap: 0.15em;">
      <img src="assets/github.svg" alt="github" style="height:1em;"/> <span>Github</span>
    </a>
    <span>|</span>
    <a href="https://huggingface.co/spaces/AI4Math/IneqMath-Leaderboard" class="custom-link" style="display: flex; align-items: center; gap: 0.15em;">
      🏆 <span>Leaderboard</span>
    </a>
    <span>|</span>
    <a href="https://ineqmath.github.io/#visualization" class="custom-link" style="display: flex; align-items: center; gap: 0.15em;">
      🔮 <span>Visualization</span>
    </a>
  </div> -->

  <!--- BADGES: START --->
[![HF license](https://img.shields.io/badge/License-CC--BY--SA--4.0-green.svg?logo=huggingface)](https://creativecommons.org/licenses/by-sa/4.0/)
[![GitHub code](https://img.shields.io/badge/Github-Code-2176BC.svg?logo=github)](https://github.com/lupantech/ineqmath)
[![Arxiv](https://img.shields.io/badge/arXiv-2506.07927-B31B1B.svg?logo=arxiv)](https://arxiv.org/abs/2506.07927)
[![HF Paper](https://img.shields.io/badge/Huggingface-Paper-FFD21E.svg?logo=huggingface)](https://huggingface.co/papers/2506.07927)
[![Evaluation Platform](https://img.shields.io/badge/Evaluation-Platform-FFD21E.svg?logo=huggingface)](https://huggingface.co/spaces/AI4Math/IneqMath-Leaderboard)
[![Website](https://img.shields.io/badge/Website-IneqMath-2176BC?logo=GoogleChrome)](https://ineqmath.github.io/)
[![Leaderboard](https://img.shields.io/badge/Leaderboard-IneqMath-FFD21E?logo=Hoppscotch)](https://ineqmath.github.io/#leaderboard)
[![Visualization](https://img.shields.io/badge/Visualization-IneqMath-D03C36?logo=Vega)](https://ineqmath.github.io/#visualization)
[![Coverage](https://img.shields.io/badge/Coverage-IneqMath-2176BC.svg?logo=x)](https://x.com/lupantech/status/1932866286427779586)


<!--- BADGES: END --->

</div>


# Introduction
Inequality proving, crucial across diverse scientific and mathematical fields, tests advanced reasoning skills such as discovering tight bounds and strategically applying theorems. This makes it a distinct and demanding frontier for large language models (LLMs), offering insights beyond general mathematical problem-solving. Progress in this area is hampered by existing datasets that are often scarce, synthetic, or rigidly formal. We address this by proposing an **informal yet verifiable** task formulation, recasting inequality proving into two automatically checkable subtasks: **bound estimation** and **relation prediction**.

Building on this, we release <b><span style="color:#103756;">Ineq</span><span style="color:#D03C36;">Math</span></b>, an expert-curated dataset of Olympiad-level inequalities, including a test set and training corpus enriched with **step-wise solutions and theorem annotations**. We also develop a novel **LLM-as-judge evaluation framework**, combining a *final-answer* judge with four *step-wise* judges designed to detect common reasoning flaws.

A systematic evaluation of 29 leading LLMs on <b><span style="color:#103756;">Ineq</span><span style="color:#D03C36;">Math</span></b> reveals a surprising reality: even top models like o1 achieve less than 10% overall accuracy under step-wise scrutiny; this is a drop of up to 65.5% from their accuracy when considering only final answer equivalence. This discrepancy exposes **fragile deductive chains and a critical gap for current LLMs between merely finding an answer and constructing a rigorous proof**. **Scaling model size and increasing test-time computation** yield limited gains in overall proof correctness. Instead, our findings highlight promising research directions such as **theorem-guided reasoning and self-refinement**.

# Dataset Examples
Below are training and testing examples from <b><span style="color:#103756;">Ineq</span><span style="color:#D03C36;">Math</span></b>. Each problem belongs to one of two automatically checkable subtasks: **bound estimation** or **relation prediction**. Each training problem includes **step-wise solutions**, with up to four solutions per problem, and 76.8% (962 problems) are annotated with **relevant theorems**. The test problems are each crafted and reviewed by **IMO-level medalists** to ensure both originality and difficulty.

Training examples of <b><span style="color:#103756;">Ineq</span><span style="color:#D03C36;">Math</span></b>:
<div align="center">
    <img src="assets/examples/train_bound_example.png" width="650" alt="Train Bound Example">
    <img src="assets/examples/train_relation_example.png" width="650" alt="Train Relation Example">
</div>


Testing examples of <b><span style="color:#103756;">Ineq</span><span style="color:#D03C36;">Math</span></b>:


<div align="center">
    <img src="assets/examples/test_bound_example.png" width="650" alt="Test Bound Example">
    <img src="assets/examples/test_relation_example.png" width="650" alt="Test Relation Example">
</div>

# Dataset Overview
The <b><span style="color:#103756;">Ineq</span><span style="color:#D03C36;">Math</span></b> dataset comprises 200 test problems for benchmarking, 100 development problems with public ground truth, and 1,252 training problems split evenly between **bound estimation** and **relation prediction** tasks as shown in the table below. The dataset also features 83 named theorems across 29 categories, with their distribution illustrated in the figure below.
<center>
  <table 
    align="center" 
    width="60%" 
    border="1" 
    cellspacing="0" 
    cellpadding="6"
    style="width:60%; table-layout: fixed; border-collapse: collapse; text-align: center;">
    <colgroup>
      <col width="64%">
      <col width="12%">
      <col width="12%">
      <col width="12%">
    </colgroup>
    <thead>
      <tr>
        <th style="text-align:left;">Statistic</th>
        <th>Number</th>
        <th>Bnd.</th>
        <th>Rel.</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="text-align:left;"><b>Theorem categories</b></td>
        <td>29</td>
        <td>–</td>
        <td>–</td>
      </tr>
      <tr style="border-bottom:2px solid #000;">
        <td style="text-align:left;"><b>Named theorems</b></td>
        <td>83</td>
        <td>–</td>
        <td>–</td>
      </tr>
      <tr>
        <td style="text-align:left;"><b>Training problems (for training)</b></td>
        <td>1252</td>
        <td>626</td>
        <td>626</td>
      </tr>
      <tr>
        <td style="text-align:left;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- With theorem annotations</td>
        <td>962</td>
        <td>482</td>
        <td>480</td>
      </tr>
      <tr>
        <td style="text-align:left;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- With solution annotations</td>
        <td>1252</td>
        <td>626</td>
        <td>626</td>
      </tr>
      <tr>
        <td style="text-align:left;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Avg. solutions per problem</td>
        <td>1.05</td>
        <td>1.06</td>
        <td>1.05</td>
      </tr>
      <tr style="border-bottom:2px solid #000;">
        <td style="text-align:left;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Max solutions per problem</td>
        <td>4</td>
        <td>4</td>
        <td>4</td>
      </tr>
      <tr>
        <td style="text-align:left;"><b>Dev problems (for development)</b></td>
        <td>100</td>
        <td>50</td>
        <td>50</td>
      </tr>
      <tr>
        <td style="text-align:left;"><b>Test problems (for benchmarking)</b></td>
        <td>200</td>
        <td>96</td>
        <td>104</td>
      </tr>
    </tbody>
  </table>
</center>
<br>

<div align="center">

  <img src="./assets/dataset/theorem_category_pie_chart.png" alt="IneqMath Logo" width="520"/>

</div>

The table below compares datasets for inequalities and theorem proving. <b><span style="color:#103756;">Ineq</span><span style="color:#D03C36;">Math</span></b> provides expert-annotated training and test/dev sets, featuring high-quality named theorems and step-wise solutions for model development. Unlike prior datasets that use synthesis or autoformalization, <b><span style="color:#103756;">Ineq</span><span style="color:#D03C36;">Math</span></b> presents problems in informal language across both multiple-choice (MC) and open-ended (Open) formats, and employs LLM-as-judge for evaluation.

<div align="center">
<img src="assets/dataset/dataset_comparison.png" width="90%">
</div>


# Dataset Usage
## Load dataset in Python
You can download this dataset by the following command (make sure that you have installed [Huggingface Datasets](https://huggingface.co/docs/datasets/quickstart)):

```python
from datasets import load_dataset
dataset = load_dataset("AI4Math/IneqMath")
```

Here are some examples of how to access the downloaded dataset:

```python
# print the first data on the training set
print(dataset["train"][0])
# print the first data on the test set
print(dataset["test"][0])
# print the first data on the dev set
print(dataset["dev"][0])
```

## Download json form dataset
You can also download the whole json form dataset by running the following command:
```shell
wget https://huggingface.co/datasets/AI4Math/IneqMath/resolve/main/data/json/all.tar.gz
```
Then, please uncommpress the file:
```shell
tar -zxvf all.tar.gz
cd json
```

The file structure of the uncommpressed file is as follows:
<details>
<summary>
Click to expand the file structure
</summary>

```
json
├── train.json # Train set
├── test.json # Test set
├── dev.json # Dev set
└── theorems.json # Theorems set

```

</details>

## Data Format
The dataset is provided in json format and contains the following attributes:

```json
{
    "data_id": [integer] The ID of the data of each split,
    "problem": [string] The question text,
    "type": [string] The type of question: ‘relation’ or 'bound',
    "data_split": [string] Data split: 'train', 'test' or 'dev',
    "answer": [string] The correct answer of the problem,
    "solution": [string] Step by step solution of the problem,
    "theorems": [Dictionary] A dictionary of manually annotated theorems that are relevant or expected to be used in solving the problem. Each theorem has key and value shown below:              
        Theorem_id: [string] The ID of the theorem. For example, 'Theorem_1' is the ID for the first theorem
            {
            "Nickname": [list] A list of nicknames of the theorem,
            "Theorem": [string] The content of the theorem,
            "Theorem_Category": [string] the category of the theorem
            }
    "choices": [list] A list of choices of the multi-choice relation problem. If the problem type is 'bound', choices would be null.
}
```

The theorem set is provided in json format and contains the following attributes:
```json
Theorem_id: [string] The ID of the theorem. For example, 'Theorem_1' is the ID for the first theorem
            {
            "Nickname": [list] A list of nicknames of the theorem,
            "Theorem": [string] The content of the theorem,
            "Theorem_Category": [string] the category of the theorem
            }
```

# Fine-grained Informal Judges

Traditional evaluation methods fall short in this setting: expert annotation is accurate but prohibitively labor-intensive, while automated techniques such as string matching or value equivalence fail to capture step-by-step correctness—an essential aspect of inequality problem solving. To evaluate the correctness of <b><span style="color:#103756;">Ineq</span><span style="color:#D03C36;">Math</span></b> solutions, we propose a fine-grained **LLM-as-judge** framework, consisting of a **final-answer judge** for verifying the predicted answer and four specialized **step-wise judges** targeting common reasoning flaws. A solution is deemed correct **overall** only if it passes all five judges. As shown in the following table and confusion matrix, these judges achieve strong alignment with human annotations (F1 = 0.93), providing a scalable yet reliable alternative to manual evaluation.

<div align="center">

  <img src="./assets/results_figures/confusion_matrix_judge_metrix.png" alt="judge_confusion_matrix" width="800"/>
  <img src="./assets/results_figures/table_judge_metrics.png" alt="table_judge_matrix" width="650"/>

</div>

# Results of leading LLMs
This table shows the **Final-answer accuracy** versus **overall accuracy** for leading LLMs across different categories on the <b><span style="color:#103756;">Ineq</span><span style="color:#D03C36;">Math</span></b>benchmark of Olympiad-level inequality problems. Overall accuracy, measuring both answer correctness and step soundness, is substantially lower than final-answer accuracy for all model types. This highlights a critical gap: while LLMs may find correct final answers to these inequality problems, their reasoning is often unsound. Each model used its optimal maximal tokens.

<div align="center">

  <img src="./assets/results_figures/key_results_figure.png" alt="judge_confusion_matrix" width="800"/>

</div>


# Scaling law in model size
The following two figures show how <em>final-answer accuracy</em> (which evaluates only the correctness of the final predicted answer) and <em>overall accuracy</em> (which requires both a correct answer and valid intermediate reasoning steps) scales with model size for LLMs.

The figure below shows how final-answer accuracy (which evaluates only the correctness of the final predicted answer) scales with model size for LLMs. As model size increases, we observe a steady improvement in answer accuracy, reflecting an empirical scaling law that larger models are better at inferring correct bounds and inequality relationships.
<div align="center">

  <img src="./assets/results_figures/scaling_law_model_size_answer_acc_log_all.png" alt="scaling_curve_answer_acc" width="700"/>

</div>

 However, the trend for answer accuracy does not hold well when considering overall accuracy—which requires both a correct answer and valid intermediate reasoning steps—as shown in the figure below. In this case, the scaling curve flattens, indicating that increased model size alone is insufficient to eliminate step-by-step reasoning errors.

<div align="center">
  <img src="./assets/results_figures/scaling_law_model_size_overall_acc_log_all.png" alt="scaling_curve_overall_acc" width="700"/>

</div>

# Retrieving relevant theorems as hints
As shown in the figure, providing one or two such theorems decreases overall accuracy for weaker models (e.g., Grok 3 mini, o3-mini, o4-mini), likely due to misapplication or distraction by potentially irrelevant information. Conversely, stronger models like Gemini 2.5 Pro benefit from these hints, suggesting advanced reasoning is crucial to effectively use such guidance. These results underscore the potential of theorem-guided reasoning but also highlight the critical need for more sophisticated theorem retrieval mechanisms (e.g., RAG) to reliably enhance LLM performance in inequality proving.

<div align="center">
<img src="assets/results_figures/theorem_as_hints.png" width="65%">
</div>

# Self-improvement via critic as feedback
As the figure shows, self-critique consistently improves performance—e.g., Gemini 2.5 Pro's overall accuracy rises from 43% to 48%. This upward trend underscores self-critique as a promising, supervision-free method to enhance logical rigor and solution quality of LLMs in inequality reasoning.

<div align="center">
<img src="assets/results_figures/self_critic.png" width="65%">
</div>

# License

The new contributions to our dataset are distributed under the [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license.

The copyright of the images and the questions belongs to the original authors. Alongside this license, the following conditions apply:

- **Purpose:** The test split was primarily designed for use as a test set.
- **Commercial Use:** The test split can be used commercially as a test set, but using it as a training set is prohibited. By accessing or using this dataset, you acknowledge and agree to abide by these terms in conjunction with the [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license.

# Citation

If you use the <b><span style="color:#103756;">Ineq</span><span style="color:#D03C36;">Math</span></b> dataset in your work, please kindly cite the paper using this BibTeX:

```
@inproceedings{lu2025solving,
  title={Solving Inequality Proofs with Large Language Models},
  author={Lu, Pan and Sheng, Jiayi and Lyu, Luna and Jin, Jikai and Xia, Tony and Gu, Alex and Zou, James},
  booktitle={The 39th Conference on Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```