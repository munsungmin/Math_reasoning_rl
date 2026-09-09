# Math_reasoning_rl

Qwen2.5-1.5B math reasoning experiments with verl, FSDP/Megatron launch scripts,
Conda environment records, and prepared math datasets.

## Clone and restore dependency changes

```bash
git clone --recurse-submodules https://github.com/munsungmin/Math_reasoning_rl.git
cd Math_reasoning_rl
python script/apply_dependency_patches.py
```

For an existing clone: `git submodule update --init --recursive` first.
The parent repository pins these upstream commits:

| Submodule | Commit | Purpose |
|---|---|---|
| train_rl/verl | 89dad2d7 | Original upstream checkout |
| train_rl/verl-cu121 | 070ed6ac | CUDA 12.1 compatible checkout, v0.3.0.post1 |
| train_rl/apex | b496d85f | Apex CUDA extension source |

Local source edits are preserved in `train_rl/patches/`, including the existing
import-path changes and the three CUDA compatibility edits. Submodule commits
alone do not contain these edits; apply the patches to restore this snapshot.
The helper skips patches that are already applied and refuses unexpected bases
or conflicting changes. Patched submodules normally appear as modified in
`git status`. Their source contents have not been reset or deleted.

The current import-path edits include `RL.verl...` and `train_rl.apex...` imports.
They were preserved as found, but were not validated as runnable in this layout.
The earlier successful FSDP smoke test predates the directory move and these edits.

## Files

- `dataset/`: GSM8K, MATH, MATH levels 3–5, MATH-500, AIME 2024/2025, IneqMath.
  See [dataset details](dataset/README.md).
- `script/`: training launchers and reproducible dataset preparation/checks.
- `train_rl/setup-verl/`: Conda specifications, compatibility patch and smoke results.
- `train_rl/patches/`: exact local dependency changes plus base commit manifest.

Execution logs, generated outputs and Python caches are kept locally and ignored
by Git. Existing environment exports and historical reports may contain paths
from before the directory move; adapt them to the checkout location when installing.
