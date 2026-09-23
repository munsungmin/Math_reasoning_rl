"""Stable identities for generated and classified predictions."""


def prediction_key(row):
    # Each collection belongs to one policy/evaluation; callers must not mix runs.
    return (
        row.get("dataset", ""),
        row.get("split", ""),
        str(row["sample_id"]),
        int(row.get("sample_index", 0)),
    )


class PredictionData:
    """Deduplicate identical distributed padding; reject conflicting predictions."""

    def __init__(self):
        self.records = {}

    def add(self, rows):
        added = []
        for row in rows:
            key = prediction_key(row)
            if key in self.records:
                if self.records[key] != row:
                    raise ValueError(f"Conflicting prediction for {key}")
                continue
            self.records[key] = dict(row)
            added.append(row)
        return added
