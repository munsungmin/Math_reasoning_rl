"""Task views preserve raw prompts and keep supervised targets out of inference."""

ERROR_LABELS = (
    "Correct",
    "Mathematical_Error",
    "Logical_Inconsistency",
    "Accumulation_Error",
)


def normalize_row(row, *, dataset, split, kind="solve", classification="multi"):
    identity = row.get("sample_id", row.get("id", row.get("unique_id")))
    if identity is None:
        raise ValueError("Every record requires a stable sample_id/id/unique_id")
    result = dict(row, sample_id=str(identity), dataset=dataset, split=split)
    prompt = row.get("prompt", row.get("problem"))
    if kind == "classify":
        label = row["label"]
        if isinstance(label, str):
            if label not in ERROR_LABELS:
                raise ValueError(f"Unknown error label: {label}")
            if classification == "multi":
                label = ERROR_LABELS.index(label)
            elif classification == "binary":
                label = int(label != "Correct")
            elif classification == "single":
                label = int(label == "Logical_Inconsistency")
            else:
                raise ValueError(f"Unknown classification: {classification}")
        prompt = row.get("prompts", {}).get(classification, prompt)
        if not 0 <= int(label) < (4 if classification == "multi" else 2):
            raise ValueError(f"Label outside {classification} label space: {label}")
        result["label"] = int(label)
        result["target"] = "ABCD"[int(label)]
    if isinstance(prompt, list):
        # Preserve the legacy single-user prompt representation without extra templates.
        if len(prompt) != 1 or prompt[0].get("role") != "user":
            raise ValueError("This prompt view expects one user message")
        prompt = prompt[0]["content"]
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(f"Missing text prompt for {identity}")
    result["prompt"] = prompt
    if "answer" not in result and "reward_model" in row:
        result["answer"] = row["reward_model"]["ground_truth"]
    if kind == "solve" and "target" not in result and "solution" in row:
        result["target"] = (
            f"<reasoning>\n{row['solution'].strip()}\n</reasoning>\n<answer>{result['answer']}</answer>"
        )
    return result


def check_disjoint(*splits):
    """Legacy frozen RL contract: IDs, families and normalized questions are unique."""
    for field in ("sample_id", "family_id", "problem"):
        seen = set()
        for rows in splits:
            keys = [str(row[field]) for row in rows]
            if field == "problem":
                keys = ["".join(key.lower().split()) for key in keys]
            if len(set(keys)) != len(keys) or seen.intersection(keys):
                raise ValueError(f"Duplicate/overlapping {field} in evaluation splits")
            seen.update(keys)
