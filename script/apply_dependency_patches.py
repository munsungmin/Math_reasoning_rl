"""Restore the preserved dependency edits after cloning the submodules."""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(path, *args):
    return subprocess.run(['git', '-C', str(path), *args], capture_output=True, text=True)


if __name__ == '__main__':
    pending = []
    for item in json.loads((ROOT / 'train_rl/patches/manifest.json').read_text()):
        path, patch = ROOT / item['path'], ROOT / item['patch']
        head = run(path, 'rev-parse', 'HEAD')
        if head.returncode or head.stdout.strip() != item['commit']:
            raise SystemExit(f"Unexpected base in {path}; initialize the pinned submodule first.")
        if run(path, 'apply', '--reverse', '--check', str(patch)).returncode == 0:
            print(f"Already applied: {item['path']}")
            continue
        check = run(path, 'apply', '--check', str(patch))
        if check.returncode:
            raise SystemExit(f'Patch conflicts in {path}:\n{check.stderr}')
        pending.append((path, patch))
    for path, patch in pending:
        result = run(path, 'apply', str(patch))
        if result.returncode:
            raise SystemExit(result.stderr)
        print(f'Applied: {patch.name}')
