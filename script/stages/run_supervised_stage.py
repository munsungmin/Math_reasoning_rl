"""Own-process launcher for one gated Lightning C/S stage."""

import argparse
import json
import hashlib
import os
import subprocess
import time
import signal
import shutil
from pathlib import Path

ROOT = Path("/home/sungmin/math_reasoning")
BASE = Path("/data/sungmin/math_reasoning")
PY = BASE / "envs/math-supervised/bin/python"
p = argparse.ArgumentParser()
p.add_argument("--arm", required=True)
p.add_argument("--seed", type=int, required=True)
p.add_argument(
    "--kind",
    choices=[
        "classify_multi",
        "classify_binary",
        "classify_single",
        "solve_sft",
    ],
    required=True,
)
p.add_argument("--order", type=int, default=1)
p.add_argument("--parent")
p.add_argument("--chain-root")
a = p.parse_args()
assert json.loads(
    (BASE / "artifacts/reward_gate/20260910-025950/gate.json").read_text()
)["passed"]
if a.parent:
    assert (Path(a.parent) / "adapter_model.safetensors").is_file()
code = ROOT / "script/stages/train_supervised_stage.py"
spec = dict(
    arm=a.arm,
    seed=a.seed,
    kind=a.kind,
    parent=a.parent,
    epochs=2 if a.kind == "solve_sft" else 3,
    code_sha256=hashlib.sha256(code.read_bytes()).hexdigest(),
    data_manifest_sha256=hashlib.sha256(
        (
            BASE
            / "artifacts/math_error_transfer/frozen/v1_20260910/manifest.json"
        ).read_bytes()
    ).hexdigest(),
)
hash8 = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[
    :8
]
rid = time.strftime("run-%Y%m%dT%H%M%S") + "-" + hash8
run = (
    Path(a.chain_root)
    if a.chain_root
    else BASE
    / "checkpoints/math_error_transfer/main"
    / a.arm
    / f"seed-{a.seed:04d}"
    / rid
)
stage = run / "stages" / f"{a.order:02d}_{a.kind}"
stage.mkdir(parents=True, exist_ok=False)
art = (
    BASE
    / "artifacts/math_error_transfer/main"
    / a.arm
    / f"seed-{a.seed:04d}"
    / run.name
    / f"{a.order:02d}_{a.kind}"
)
art.mkdir(parents=True)
spec.update(
    run_root=str(run),
    stage_root=str(stage),
    artifact_root=str(art),
    status="running",
)
(run / "run_manifest.json").write_text(json.dumps(spec, indent=2))
(stage / "parent.json").write_text(
    json.dumps(
        dict(
            parent=a.parent or "Qwen2.5-Math-1.5B@4a83ca6e",
            optimizer_reset=True,
        ),
        indent=2,
    )
)
shutil.copy2(code, art / code.name)
if shutil.disk_usage(BASE).free < 32 * 2**30:
    raise RuntimeError("Less than 32 GiB free")
for attempt in range(13):
    rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=20,
    )
    busy = [
        i
        for i, mem, util in (map(int, l.split(",")) for l in rows.splitlines())
        if i in [2, 3, 4, 5, 6] and (mem > 100 or util)
    ]
    if not busy:
        break
    if attempt == 12:
        raise RuntimeError(
            f"Training GPUs occupied: {busy}; no process stopped"
        )
    time.sleep(5)
env = dict(
    os.environ,
    PYTHONUTF8="1",
    CUDA_VISIBLE_DEVICES="2,4,5,6",
    CUDA_MODULE_LOADING="LAZY",
    HF_HUB_OFFLINE="1",
    NCCL_P2P_DISABLE="1",
    NCCL_IB_DISABLE="1",
    OMP_NUM_THREADS="1",
    STAGE_ROOT=str(stage),
    STAGE_ARTIFACT=str(art),
    STAGE_KIND=a.kind,
    STAGE_SEED=str(a.seed),
)
if a.parent:
    env["STAGE_PARENT"] = a.parent
train = None
try:
    with (art / "train.log").open("w") as log:
        train = subprocess.Popen(
            [
                str(PY),
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=4",
                str(code),
            ],
            env=env,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        code = train.wait(timeout=18000)
        if code:
            raise RuntimeError(
                f"Supervised process failed ({code}); see {art}/train.log"
            )
    result = json.loads((stage / "checkpoint_manifest.json").read_text())
    assert (
        result["completed"]
        and result["reload_verified"]
        and result["next_stage_gate"]
    ), "Supervised learning gate failed; dependent RL not started"
    spec.update(
        status="completed",
        final_export=str(stage / "exports/final"),
        finished=time.time(),
    )
except Exception as e:
    spec.update(status="failed", error=repr(e))
    raise
finally:
    if train is not None and train.poll() is None:
        os.killpg(train.pid, signal.SIGTERM)
        try:
            train.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(train.pid, signal.SIGKILL)
    (run / "run_manifest.json").write_text(json.dumps(spec, indent=2))
    (stage / "stage_manifest.json").write_text(json.dumps(spec, indent=2))
print("SUPERVISED_STAGE_COMPLETED", json.dumps(spec), flush=True)
