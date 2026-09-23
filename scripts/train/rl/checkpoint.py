"""Extracted native verl checkpoint; legacy run semantics are preserved."""

import json
import os
import re
import shutil
import uuid

WORLD_SIZE = 4


def validate_checkpoint(root, step, world_size=WORLD_SIZE):
    """The native completion marker is written only after all ranks finish saving."""
    checkpoint = root / f"global_step_{step}"
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise RuntimeError(f"실제 checkpoint 폴더가 아닙니다: {checkpoint}")
    required = [
        "data.pt",
        "actor/fsdp_config.json",
        "actor/lora_train_meta.json",
        "actor/huggingface/config.json",
        "actor/huggingface/tokenizer_config.json",
    ]
    required += [
        f"actor/{kind}_world_size_{world_size}_rank_{rank}.pt"
        for rank in range(world_size)
        for kind in ("model", "optim", "extra_state")
    ]
    for name in required:
        path = checkpoint / name
        if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
            raise RuntimeError(f"불완전한 checkpoint: {path}")
    fsdp_config = json.loads((checkpoint / "actor/fsdp_config.json").read_text())
    if fsdp_config["world_size"] != world_size:
        raise RuntimeError(f"Checkpoint GPU world size가 {world_size}가 아닙니다.")
    return checkpoint


def publish_last(root, world_size=WORLD_SIZE, keep_last_only=True):
    """Publish the complete save first; delete only older saves in this run."""
    marker = root / "latest_checkpointed_iteration.txt"
    if not marker.exists():
        return None
    value = marker.read_text().strip()
    if not value:  # Native writer briefly truncates the marker before writing.
        return None
    step = int(value)
    last = root / "last"
    if last.exists() and not last.is_symlink():
        raise RuntimeError(f"last가 symlink가 아니므로 교체하지 않습니다: {last}")
    if last.is_symlink() and last.resolve().parent != root.resolve():
        raise RuntimeError("last가 현재 실행 밖을 가리킵니다.")
    checkpoint = validate_checkpoint(root, step, world_size)
    changed = not last.is_symlink() or last.resolve() != checkpoint.resolve()
    if changed:
        temporary = root / f".last-{uuid.uuid4().hex}"
        try:
            temporary.symlink_to(checkpoint.name, target_is_directory=True)
            os.replace(temporary, last)
        finally:
            temporary.unlink(missing_ok=True)
    for previous in root.iterdir():
        match = re.fullmatch(r"global_step_(\d+)", previous.name)
        if (
            keep_last_only
            and match
            and int(match[1]) < step
            and previous.is_dir()
            and not previous.is_symlink()
        ):
            shutil.rmtree(previous)
    return step if changed else None
