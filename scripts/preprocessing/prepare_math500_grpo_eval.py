"""Freeze selected original MATH-500 levels, without solution text."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--four-prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--levels", type=int, nargs="+", default=[3, 4, 5])
    args = parser.parse_args()
    source = [json.loads(s) for s in args.source.read_text().splitlines()]
    four = {
        r["sample_id"]: r
        for r in map(json.loads, args.four_prompts.read_text().splitlines())
    }
    assert len(source) == 500 and len({r["unique_id"] for r in source}) == 500
    assert len(four) == 4
    assert len(set(args.levels)) == len(args.levels) and set(args.levels) <= {
        1,
        2,
        3,
        4,
        5,
    }
    selected = [r for r in source if r["level"] in args.levels]
    expected = {1: 43, 2: 90, 3: 105, 4: 128, 5: 134}
    assert Counter(r["level"] for r in selected) == {
        level: expected[level] for level in args.levels
    }
    prompts = []
    metadata = []
    for row in selected:
        prompt = (
            "You are a math assistant. Solve the problem step by step inside <reasoning> tags, "
            "then give your final answer inside <answer> tags.\n\nQuestion: "
            + row["problem"]
            + "\n\nAnswer:"
        )
        item = dict(sample_id=row["unique_id"], prompt=prompt, answer=row["answer"])
        if item["sample_id"] in four:
            assert item == four[item["sample_id"]], (
                "Original overfit prompt or answer changed"
            )
        prompts.append(item)
        metadata.append(
            dict(
                sample_id=row["unique_id"],
                level=row["level"],
                subject=row["subject"],
                used_for_four_problem_rl=row["unique_id"] in four,
            )
        )
    assert sum(m["used_for_four_problem_rl"] for m in metadata) == 4
    args.out.mkdir(parents=True, exist_ok=False)
    prompt_path = args.out / "prompts.jsonl"
    prompt_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in prompts)
    )
    manifest = dict(
        source=str(args.source),
        source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        source_count=500,
        selected_count=len(prompts),
        level_counts=dict(Counter(m["level"] for m in metadata)),
        trained_four_ids=list(four),
        excluding_trained_four_count=len(selected) - 4,
        prompts_sha256=hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
        solution_text_in_evaluation_input=False,
        records=metadata,
    )
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {k: v for k, v in manifest.items() if k != "records"}, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
