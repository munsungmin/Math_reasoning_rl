"""Mix all arms/seeds into opaque judge batches; keep the mapping separate."""

import json
import random
import hashlib
from pathlib import Path

BASE = Path("/data/sungmin/math_reasoning")
ROOT = Path("/home/sungmin/math_reasoning")
OUT = BASE / "reports/math_error_transfer/judge_batches"
OUT.mkdir(parents=True, exist_ok=False)
rows = []
for f in sorted(
    (BASE / "artifacts/math_error_transfer/evaluation").glob(
        "*/*/blind_judge_input.jsonl"
    )
):
    rows.extend(json.loads(l) for l in f.read_text().splitlines())
assert len({r["trajectory_id"] for r in rows}) == len(rows)
random.Random(20260909).shuffle(rows)
files = []
for start in range(0, len(rows), 5):
    batch = rows[start : start + 5]
    name = (
        "batch_"
        + hashlib.sha256(
            "".join(r["trajectory_id"] for r in batch).encode()
        ).hexdigest()[:12]
        + ".jsonl"
    )
    (OUT / name).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in batch),
        encoding="utf-8",
    )
    files.append(dict(file=name, trajectories=len(batch)))
(OUT / "manifest.json").write_text(
    json.dumps(
        dict(
            trajectories=len(rows),
            batches=files,
            arm_or_seed_identifiers_in_inputs=False,
            annotation_status="pending",
            segmenter="blank-paragraph-v1; raw text preserved",
        ),
        indent=2,
    )
)
(OUT / "JUDGE_INSTRUCTIONS.md").write_bytes(
    (
        ROOT / "experiments/math_error_transfer/JUDGE_INSTRUCTIONS.md"
    ).read_bytes()
)
(OUT / "judgement.schema.json").write_bytes(
    (
        ROOT / "experiments/math_error_transfer/judgement.schema.json"
    ).read_bytes()
)
print(OUT)
