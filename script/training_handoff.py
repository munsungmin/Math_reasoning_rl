"""Explicitly defer a scheduled evaluation when a subsequent RL run is queued."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from math500_eval_scope import write_json


HANDOFF_EXIT = 75


def defer_for_training(args):
    request_path = args.out.parent / 'next_training_request.json'
    if not request_path.exists():
        return
    request = json.loads(request_path.read_text())
    if args.out.name != 'evaluation_last':
        return
    assert request['version'] == 1 and request['mode'] == 'train_then_evaluate'
    assert Path(request['source_run']).resolve() == args.out.parent.resolve()
    assert (args.adapter / 'adapter_model.safetensors').is_file()
    write_json(args.out.parent / 'training_handoff.json', dict(
        status='evaluation_deferred_for_requested_training',
        at=datetime.now(timezone.utc).isoformat(), exported_adapter=str(args.adapter),
        adapter_sha256=hashlib.sha256((args.adapter / 'adapter_model.safetensors').read_bytes()).hexdigest(),
        request_sha256=hashlib.sha256(request_path.read_bytes()).hexdigest(),
        next_run=request['next_run'], expected_exit_code=HANDOFF_EXIT))
    print('Evaluation deferred: the user requested the next pure-RL training stage first.', flush=True)
    raise SystemExit(HANDOFF_EXIT)
