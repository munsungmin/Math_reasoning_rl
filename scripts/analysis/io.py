"""Read common artifacts without loading a model, backend, or plotting library."""

import json
from pathlib import Path

from scripts.utils.io import read_rows


def load_run(directory):
    directory = Path(directory)
    result = {"path": str(directory)}
    for name in ("manifest", "result", "summary"):
        path = directory / f"{name}.json"
        if path.exists():
            result[name] = json.loads(path.read_text())
    result["observations"] = (
        read_rows(directory / "observations/metrics.jsonl")
        if (directory / "observations/metrics.jsonl").exists()
        else []
    )
    result["evaluations"] = {
        str(p.parent.relative_to(directory)): json.loads(p.read_text())
        for folder in ("val", "evaluations")
        for p in (directory / folder).rglob("summary.json")
    }
    return result
