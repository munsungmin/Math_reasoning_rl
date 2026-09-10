#!/usr/bin/env bash
set -euo pipefail
TASK_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_PY=/home/sungmin/.conda/envs/grpo-lora/bin/python
TASK_RUN=/data/sungmin/math_reasoning/checkpoints/math_error_transfer/main/E1/seed-0017/run-20260910T030734-8eaaf25e
cd "$TASK_REPO"
case "${1:-status}" in
  status) exec "$TASK_PY" script/math_run_status.py ;;
  logs)
    exec "$TASK_PY" script/math_run_status.py --log /data/sungmin/math_reasoning/artifacts/math_error_transfer/main/E1/seed-0017/run-20260910T030734-8eaaf25e/math500_level3_5/train_resume.log "${@:2}"
    ;;
  resume)
    if "$TASK_PY" -c "import sys; sys.path.insert(0,'script'); from math_run_status import active_jobs; sys.exit(0 if active_jobs() else 1)"; then
      echo '학습 프로세스가 이미 실행 중입니다. 중복 실행하지 않습니다. status 또는 logs를 사용하세요.'
      exit 0
    fi
    exec "$TASK_PY" script/run_rl_stage.py --resume "$TASK_RUN"
    ;;
  queue)
    TASK_PID_FILE=/data/sungmin/math_reasoning/artifacts/math_error_transfer/queue_20260910/pid.txt
    if test -f "$TASK_PID_FILE" && kill -0 "$(cat "$TASK_PID_FILE")" 2>/dev/null; then
      echo '순차 실행 queue가 이미 실행 중입니다.'
      exit 0
    fi
    exec "$TASK_PY" script/stages/run_math_experiment_queue.py
    ;;
  new-e1)
    if "$TASK_PY" -c "import sys; sys.path.insert(0,'script'); from math_run_status import active_jobs; sys.exit(0 if active_jobs() else 1)"; then
      echo '기존 학습이 실행 중이므로 새 실험을 시작하지 않습니다.'
      exit 0
    fi
    exec "$TASK_PY" script/run_rl_stage.py --arm E1 --seed "${2:-17}"
    ;;
  *) echo '사용법: bash script/run_math_main.sh {status|logs [--raw]|resume|queue|new-e1 [17|29]}' >&2; exit 2 ;;
esac
