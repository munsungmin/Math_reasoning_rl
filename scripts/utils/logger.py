"""Persist all results locally; optional external logging is an additional sink."""

import json
from pathlib import Path


class ResultWriter:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def log(self, values, *, step, stage, source="metric"):
        event = {"step": int(step), "stage": stage, "source": source, "values": values}
        with (self.directory / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        return event
