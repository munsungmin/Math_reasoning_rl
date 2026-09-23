"""Extend frozen split membership to all levels using a reproducible 400/50/50 split."""

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()]


def family_id(problem):
    text = "".join(unicodedata.normalize("NFKC", problem).lower().split())
    return hashlib.sha256(re.sub(r"\d+(?:\.\d+)?", "#", text).encode()).hexdigest()


def take_stratified(pool, count, seed):
    counts = Counter(r["level"] for r in pool)
    quotas = {level: count * n // len(pool) for level, n in counts.items()}
    order = sorted(
        counts, key=lambda level: (-(count * counts[level] % len(pool)), level)
    )
    for level in order[: count - sum(quotas.values())]:
        quotas[level] += 1
    selected = []
    for level in sorted(counts):
        candidates = sorted(
            (r for r in pool if r["level"] == level),
            key=lambda r: hashlib.sha256(
                f"{seed}|{r['sample_id']}".encode()
            ).hexdigest(),
        )
        selected += candidates[: quotas[level]]
    ids = {r["sample_id"] for r in selected}
    return selected, [r for r in pool if r["sample_id"] not in ids]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--prior-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw = read_rows(args.source)
    assert len(raw) == len({r["unique_id"] for r in raw}) == 500
    records = {}
    for r in raw:
        prompt = (
            "You are a math assistant. Solve the problem step by step inside <reasoning> tags, "
            "then give your final answer inside <answer> tags.\n\nQuestion: "
            + r["problem"]
            + "\n\nAnswer:"
        )
        records[r["unique_id"]] = dict(
            sample_id=r["unique_id"],
            family_id=family_id(r["problem"]),
            problem=r["problem"],
            prompt=prompt,
            answer=r["answer"],
            level=r["level"],
            subject=r["subject"],
        )
    assert len({r["family_id"] for r in records.values()}) == 500
    protocol = json.loads((args.prior_run / "protocol.json").read_text())
    splits = {}
    for name, source in protocol["source_files"].items():
        previous = read_rows(source)
        for row in previous:
            current = records[row["sample_id"]]
            assert all(
                row[k] == current[k] for k in ("prompt", "answer", "problem", "level")
            )
        splits[name] = [records[r["sample_id"]] for r in previous]
    assigned = {r["sample_id"] for rows in splits.values() for r in rows}
    assert len(assigned) == 331
    remainder = [r for i, r in records.items() if i not in assigned]
    extra_validation, remainder = take_stratified(remainder, 20, "validation-17")
    extra_test, remainder = take_stratified(remainder, 13, "test-17")
    splits["train"] += remainder
    splits["validation"] += extra_validation
    splits["test"] += extra_test
    assert {k: len(v) for k, v in splits.items()} == {
        "train": 400,
        "validation": 50,
        "test": 50,
    }
    args.out.mkdir(parents=True, exist_ok=False)
    for name, rows in splits.items():
        (args.out / f"{name}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        )
    prompts = [
        dict(sample_id=r["sample_id"], prompt=r["prompt"], answer=r["answer"])
        for r in records.values()
    ]
    (args.out / "evaluation_prompts.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in prompts)
    )
    manifest = dict(
        source=str(args.source),
        source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        counts={k: len(v) for k, v in splits.items()},
        level_counts={
            k: dict(Counter(r["level"] for r in v)) for k, v in splits.items()
        },
        split_ids={k: [r["sample_id"] for r in v] for k, v in splits.items()},
        original_membership_preserved=protocol["split_ids"],
        allocation="Retain original 264/30/37; allocate remaining169 by level: train136, validation20, test13; SHA256 seed17 ordering",
        family_method="SHA256 of NFKC/lowercase/no-whitespace text with numeric literals replaced by #; all500 signatures distinct",
        solution_text_present=False,
        records=[
            dict(sample_id=r["sample_id"], level=r["level"], subject=r["subject"])
            for r in records.values()
        ],
        file_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in args.out.glob("*.jsonl")
        },
    )
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in (
                    "counts",
                    "level_counts",
                    "source_sha256",
                    "solution_text_present",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
