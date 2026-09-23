"""Report full levels 3-5 accuracy and accuracy excluding the four RL prompts."""

import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.data / "manifest.json").read_text())
    meta = {r["sample_id"]: r for r in manifest["records"]}
    prompts = {
        r["sample_id"]: r
        for r in map(json.loads, (args.data / "prompts.jsonl").read_text().splitlines())
    }
    rows = [
        json.loads(s)
        for s in (args.evaluation / "answers.jsonl").read_text().splitlines()
    ]
    settings = json.loads((args.evaluation / "summary.json").read_text())
    assert len(rows) == len(meta) == len({r["sample_id"] for r in rows}) == 367
    assert settings["n_per_problem"] == 1 and settings["temperature"] == 0
    for row in rows:
        assert row["input"] == prompts[row["sample_id"]]["prompt"]
        assert row["answer"] == prompts[row["sample_id"]]["answer"]
        row.update(meta[row["sample_id"]])

    def score(items):
        correct = sum(r["correct"] for r in items)
        return dict(
            correct=correct,
            samples=len(items),
            accuracy=correct / len(items),
            length_limit_count=sum(r["hit_length_limit"] for r in items),
            extraction_status_counts=dict(
                Counter(r["extraction_status"] for r in items)
            ),
        )

    unseen = [r for r in rows if not r["used_for_four_problem_rl"]]
    report = dict(
        dataset="Original MATH-500, all levels 3-5 (367 of 500)",
        all_levels_3_5=score(rows),
        excluding_four_rl_prompts=score(unseen),
        four_rl_prompts=score([r for r in rows if r["used_for_four_problem_rl"]]),
        by_level={
            level: dict(
                all=score([r for r in rows if r["level"] == level]),
                excluding_four=score([r for r in unseen if r["level"] == level]),
            )
            for level in (3, 4, 5)
        },
        by_subject={
            subject: score([r for r in rows if r["subject"] == subject])
            for subject in sorted({r["subject"] for r in rows})
        },
        settings={
            k: settings[k]
            for k in (
                "model",
                "adapter",
                "adapter_sha256",
                "grader_sha256",
                "temperature",
                "seed",
                "max_new_tokens",
                "n_per_problem",
            )
        },
        source_sha256=manifest["source_sha256"],
        note="Exclusion refers only to this four-problem RL lineage; it is not a claim about base-model pretraining exposure.",
    )
    (args.evaluation / "benchmark_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
