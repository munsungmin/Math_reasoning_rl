"""Native verl reward entry point; grading and reward composition stay separate."""

from scripts.eval.graders import SubprocessGrader
from scripts.reasoning.reward.components import Correctness, Format, Numeric
from scripts.reasoning.reward.composition import WeightedReward
from scripts.utils.paths import REPO


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    *,
    grader_python="/data/sungmin/math_reasoning/envs/limit-rlvr-grader/bin/python",
    grader_script=REPO / "scripts/eval/graders/worker.py",
    grader_timeout=90,
    correctness_weight=2.0,
    format_weight=0.5,
    numeric_weight=0.5,
    components=None,
    weights=None,
    **kwargs,
):
    grader = SubprocessGrader(grader_python, grader_script, grader_timeout)
    judgement = grader(
        [{"dataset": data_source, "completion": solution_str, "answer": ground_truth}]
    )[0]
    if components is None:
        components = {
            "correctness": Correctness(),
            "format": Format(),
            "numeric": Numeric(),
        }
        weights = {
            "correctness": correctness_weight,
            "format": format_weight,
            "numeric": numeric_weight,
        }
    else:
        from hydra.utils import instantiate

        components = {name: instantiate(spec) for name, spec in components.items()}
    result = WeightedReward(components, weights)(solution_str, judgement)
    result["extraction_status"] = judgement.get("extraction_status", "unknown")
    if extra_info and "sample_id" in extra_info:
        result["sample_id"] = str(extra_info["sample_id"])
    return result
