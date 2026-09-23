#!/usr/bin/env bash
# Native ext/verl launcher. Edit configs/verl_math.yaml for experiment settings.
# No arguments: Hydra default config. Old start/resume/check commands also work.
set -euo pipefail
VERL_REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
VERL_PYTHON=${VERL_PYTHON:-/data/sungmin/math_reasoning/envs/verl-current/bin/python}
if [[ ! -x "$VERL_PYTHON" ]]; then
    printf 'verl Python을 찾을 수 없습니다: %s\n' "$VERL_PYTHON" >&2
    exit 1
fi
cd "$VERL_REPO"
exec "$VERL_PYTHON" main.py "$@"
