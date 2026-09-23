"""Summarize real GRPO rollouts without generating answers or changing rewards."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def read_jsonl(path):
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    result = []
    for i, line in enumerate(lines):
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
    return result


def summarize(run, reference):
    reference_rows = read_jsonl(reference)
    identities = {r["prompt"]: r["sample_id"] for r in reference_rows}
    answers = {r["prompt"]: str(r["answer"]) for r in reference_rows}
    assert len(identities) == 4
    metrics = {r["step"]: r["data"] for r in read_jsonl(run / "metrics.jsonl")}
    grouped = defaultdict(list)
    for row in read_jsonl(run / "rollouts.jsonl"):
        assert row["input"] in identities, "A rollout used a different prompt"
        assert str(row["gts"]) == answers[row["input"]], (
            "The logged question and gold answer are misaligned"
        )
        assert row["acc"] in (0, 1), "Accuracy must be binary correctness"
        grouped[row["step"]].append(row)
    completed = sorted(s for s in metrics if len(grouped[s]) == 32)
    per_step = []
    for step in completed:
        rows = grouped[step]
        by_problem = defaultdict(list)
        for row in rows:
            by_problem[identities[row["input"]]].append(row)
        assert len(by_problem) == 4 and all(len(v) == 8 for v in by_problem.values())
        m = metrics[step]
        accuracy = sum(r["acc"] for r in rows) / len(rows)
        if "train/accuracy" in m:
            assert math.isclose(accuracy, m["train/accuracy"], abs_tol=1e-7)
        per_step.append(
            dict(
                step=step,
                correct=sum(r["acc"] for r in rows),
                samples=len(rows),
                accuracy=accuracy,
                per_problem={
                    p: dict(
                        correct=sum(r["acc"] for r in rs),
                        samples=len(rs),
                        all_wrong=all(r["acc"] == 0 for r in rs),
                        zero_reward_variance=len({r["score"] for r in rs}) == 1,
                    )
                    for p, rs in sorted(by_problem.items())
                },
                metrics={
                    k: m.get(k)
                    for k in (
                        "actor/grad_norm",
                        "actor/lr",
                        "actor/entropy",
                        "actor/entropy_loss",
                        "actor/pg_loss",
                        "response_length/mean",
                        "response_length/clip_ratio",
                        "timing_s/step",
                        "training/rollout_probs_diff_mean",
                        "critic/score/mean",
                    )
                },
            )
        )

    def window(steps):
        if not steps:
            return None
        return dict(
            from_step=steps[0]["step"],
            to_step=steps[-1]["step"],
            correct=sum(s["correct"] for s in steps),
            samples=32 * len(steps),
            accuracy=sum(s["accuracy"] for s in steps) / len(steps),
            length_cap_fraction=sum(
                s["metrics"]["response_length/clip_ratio"] for s in steps
            )
            / len(steps),
            per_problem={
                p: dict(
                    correct=sum(s["per_problem"][p]["correct"] for s in steps),
                    samples=8 * len(steps),
                    accuracy=sum(s["per_problem"][p]["correct"] for s in steps)
                    / (8 * len(steps)),
                    all_wrong_groups=sum(
                        s["per_problem"][p]["all_wrong"] for s in steps
                    ),
                    zero_reward_variance_groups=sum(
                        s["per_problem"][p]["zero_reward_variance"] for s in steps
                    ),
                )
                for p in steps[0]["per_problem"]
            },
        )

    perfect_streak = 0
    for step in reversed(per_step):
        if step["accuracy"] != 1.0:
            break
        perfect_streak += 1
    return dict(
        run=str(run),
        completed_steps=len(completed),
        latest_step=max(completed, default=0),
        consecutive_perfect_steps=perfect_streak,
        first10=window(per_step[:10]),
        last5=window(per_step[-5:]),
        last10=window(per_step[-10:]),
        all_steps=window(per_step),
        per_step=per_step,
        nonfinite_metrics=[
            dict(step=s, key=k, value=v)
            for s, m in metrics.items()
            for k, v in m.items()
            if isinstance(v, (int, float)) and not math.isfinite(v)
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = summarize(args.run, args.reference)
    if args.out:
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "per_step"}, ensure_ascii=False
        )
    )
    if report["per_step"]:
        print(json.dumps(report["per_step"][-1], ensure_ascii=False))


if __name__ == "__main__":
    main()
