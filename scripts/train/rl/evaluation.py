"""Feed native prediction records to the same metrics as independent evaluation."""

from collections import Counter

from scripts.eval.pipeline import evaluate_records
from scripts.utils.io import read_rows, write_json, write_rows


def evaluate_dump(path, metrics_config, output, split):
    indices, rows = Counter(), []
    for record in read_rows(path):
        identity = str(record["sample_id"])
        row = dict(
            record,
            sample_id=identity,
            sample_index=indices[identity],
            split=split,
            correct=bool(record["acc"]),
            answer=record.get("gts"),
        )
        indices[identity] += 1
        rows.append(row)
    values = evaluate_records(rows, metrics_config)
    write_rows(output / "predictions.jsonl", rows)
    write_json(output / "summary.json", values)
    return values
