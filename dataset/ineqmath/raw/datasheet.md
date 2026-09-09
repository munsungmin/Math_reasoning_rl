## **Datasheet for the IneqMath Dataset**

### 1. **Motivation**

* **Purpose**: To benchmark and advance large language models’ (LLMs) ability to solve Olympiad-level mathematical inequality problems through informal yet verifiable reasoning.
* **Gap Addressed**: Existing datasets are either synthetic, limited in scale, or fully formalized (Lean/Isabelle), which doesn’t align well with LLMs’ informal reasoning capabilities.
* **Intended Use**: Research in AI reasoning, evaluation of LLM mathematical proficiency, training for inequality solving and theorem application.


### 2. **Composition**

* **Content**: Olympiad-level inequality problems divided into two tasks:

  * **Bound estimation**: finding extremal constants in inequalities.
  * **Relation prediction**: determining inequality relations: >, ≥, =, ≤, <.
* **Size**:

  * **Training set**: 1,252 problems.
  * **Expanded training set**: 5,181 problems.
  * **Development set**: 100 problems.
  * **Test set**: 200 problems.
* **Annotations**:
  * Step-wise solutions (up to 4 per problem).
  * 83 named theorems across 29 categories (e.g., AM-GM, Cauchy-Schwarz, Jensen’s Inequality).


### 3. **Collection Process**

* **Sources**:
  * Training problems: advanced textbooks with Olympiad-level inequality problems.
  * Test problems: novel problems designed by IMO medalists to minimize data contamination.
* **Curation**:
  * Problems rephrased by LLMs into bound/relation tasks, then reviewed by experts.
  * Test problems validated by separate expert group for solvability and difficulty.
* **Format**:
  * Natural language statements (LaTeX compatible) rather than formal proof assistants.


### 4. **Preprocessing and Labeling**

* **Rephrasing**: Original problems reformulated into subtasks while preserving reasoning steps.
* **Annotations**: Theorem references, solution steps, answers, and theorem categories provided.
* **Verification**: Automatic answer checking + step-wise evaluation via LLM-as-judge framework.


### 5. **Uses**

* **Recommended**:

  * Benchmarking LLMs on mathematical reasoning.
  * Training and evaluating informal proof generation.
  * Research in theorem retrieval and guided reasoning.
* **Discouraged**:
  * Use as test data for training.
  * Misinterpretation as a formal proof dataset.


### 6. **Distribution**

* **Access**:

  * Hugging Face: [AI4Math/IneqMath](https://huggingface.co/datasets/AI4Math/IneqMath)

* **License**: CC-BY-SA-4.0



### 7. **Maintenance**

* **Maintainers**: Authors of the paper.
* **Contact**: {jamesz, panlu}@stanford.edu




