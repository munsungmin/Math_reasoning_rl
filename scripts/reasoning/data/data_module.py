from scripts.utils.io import read_rows, sha256

from .schema import normalize_row


class ReasoningData:
    """Backend-independent data and task views; no Lightning/CUDA import."""

    def __init__(
        self,
        sources,
        name,
        kind="solve",
        classification="multi",
        max_samples=-1,
        expected_counts=None,
        levels=None,
        prompt=None,
    ):
        self.sources = dict(sources)
        self.name, self.kind, self.classification = name, kind, classification
        self.max_samples = int(max_samples)
        self.expected_counts = dict(expected_counts or {})
        self.levels = levels
        self._splits = {}

    def rows(self, split):
        if split not in self._splits:
            source = self.sources.get(split)
            if not source:
                return []
            raw = read_rows(source)
            expected = self.expected_counts.get(split)
            if expected is not None and len(raw) != expected:
                raise ValueError(
                    f"{split}: expected {expected} records, found {len(raw)}"
                )
            rows = [
                normalize_row(
                    r,
                    dataset=self.name,
                    split=split,
                    kind=self.kind,
                    classification=self.classification,
                )
                for r in raw
            ]
            if len({r["sample_id"] for r in rows}) != len(rows):
                raise ValueError(f"Duplicate sample IDs in {split}")
            if self.levels is not None and any(
                r.get("level") not in self.levels for r in rows
            ):
                raise ValueError(f"Unexpected level in {split}")
            if self.max_samples > 0 and split == "train":
                rows = rows[: self.max_samples]
            self._splits[split] = rows
        return self._splits[split]

    def validate_splits(self):
        seen = {key: set() for key in ("sample_id", "family_id", "prompt")}
        for split, source in self.sources.items():
            if not source:
                continue
            rows = self.rows(split)
            if not rows:
                raise ValueError(f"Empty configured split: {split}")
            for key in seen:
                values = {
                    str(r[key]) if key != "prompt" else "".join(r[key].lower().split())
                    for r in rows
                    if r.get(key) is not None
                }
                if seen[key] & values:
                    raise ValueError(f"Overlapping {key} across splits at {split}")
                seen[key].update(values)

    def manifest(self):
        return {
            "name": self.name,
            "kind": self.kind,
            "classification": self.classification,
            "sources": {
                k: {"path": str(p), "sha256": sha256(p)}
                for k, p in self.sources.items()
                if p
            },
        }
