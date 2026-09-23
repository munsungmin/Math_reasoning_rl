"""Wait for the current RL run, then start all-levels RL before final evaluation."""

import argparse
import fcntl
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf

import run_timed_grpo as timed
from train_verl import REPO, validate_checkpoint, write_json


def process_start(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except FileNotFoundError:
        return None


def validate_handoff(source, request):
    handoff = json.loads((source/'training_handoff.json').read_text())
    request_path = source/'next_training_request.json'
    assert handoff['request_sha256'] == hashlib.sha256(request_path.read_bytes()).hexdigest()
    state = json.loads((source/'status.json').read_text())
    assert state['status'] == 'failed' and 'exit 75' in state['error']
    assert handoff['expected_exit_code'] == 75 and handoff['next_run'] == request['next_run']
    step = state['saved_last_step']
    native = validate_checkpoint(Path(state['checkpoints']),step)
    adapter = Path(handoff['exported_adapter'])
    assert adapter.resolve() == (source/f'export_step_{step}/lora_adapter').resolve()
    assert hashlib.sha256((adapter/'adapter_model.safetensors').read_bytes()).hexdigest() == handoff['adapter_sha256']
    cfg = OmegaConf.load(Path(state['run'])/'config.yaml')
    assert cfg.algorithm.adv_estimator == 'grpo' and not cfg.distillation.enabled
    audit = json.loads((source/'training_input_audit.json').read_text())
    assert audit['no_solution_targets'] and not audit['distillation']
    return state, dict(method='pure_grpo_continuation', source_run=str(source),
                       source_native_run=state['run'], source_checkpoint=str(native),
                       source_step=step, adapter=str(adapter), adapter_sha256=handoff['adapter_sha256'],
                       source_initialization_audit=str(source/'training_input_audit.json'),
                       optimizer='Fresh Adam state after loading the completed pure-GRPO adapter',
                       evaluation_deferred=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    args=parser.parse_args()
    request=json.loads(args.request.read_text())
    assert request['version']==1 and request['mode']=='train_then_evaluate'
    source=Path(request['source_run'])
    out=Path(request['next_run'])
    assert out.is_dir() and (out/'data/manifest.json').is_file()
    assert 0 < request['hours'] <= 24
    with (out/'.coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        status=dict(status='waiting_for_current_training',pid=os.getpid(),source_run=str(source),
                    hours=request['hours'],training_output=str(out/'training'),queued_at=timed.utc())
        write_json(out/'status.json',status)
        deadline=json.loads((source/'protocol.json').read_text())['deadline_epoch']+3600
        try:
            while True:
                if timed.CANCELLED:
                    raise InterruptedError('Continuation coordinator cancelled')
                alive=process_start(request['source_supervisor_pid'])==request['source_supervisor_start']
                if not alive:
                    if (source/'training_handoff.json').exists():
                        break
                    raise RuntimeError('Source supervisor exited without the expected training handoff')
                if time.time()>deadline:
                    raise TimeoutError('Source training did not hand off within its deadline plus one hour')
                status.update(updated_at=timed.utc())
                write_json(out/'status.json',status)
                time.sleep(10)
            previous, initialization=validate_handoff(source,request)
            write_json(out/'initialization.json',initialization)
            # Preserve the old controller's expected exit, and label the logical
            # stage as complete rather than pretending an evaluation happened.
            write_json(source/'supervisor_exit_for_handoff.json',previous)
            previous.update(status='training_complete_evaluation_deferred',
                            note='Training completed; evaluation was intentionally deferred for the user-requested next RL stage.',
                            next_training_run=str(out),updated_at=timed.utc())
            previous.pop('error',None)
            write_json(source/'status.json',previous)
            timed.wait_for_free_gpus([2,3,4,5,6])
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'','WANDB_DISABLE_GIT':'true',
                 'MATH_CONTINUATION_DATA':str(out/'data'),
                 'MATH_CONTINUATION_ADAPTER':initialization['adapter']}
            status.update(status='running_next_training',started_at=timed.utc(),
                          source_step=initialization['source_step'],adapter=initialization['adapter'])
            write_json(out/'status.json',status)
            timed.run_stage([sys.executable,str(REPO/'script/run_timed_grpo.py'),
                             '--config',request['config'],'--out',str(out/'training'),
                             '--hours',str(request['hours']),'--data-manifest',str(out/'data/manifest.json'),
                             '--initialization-provenance',str(out/'initialization.json')],
                            env,out/'continuation.log',(request['hours']+3)*3600)
            final=json.loads((out/'training/status.json').read_text())
            assert final['status']=='complete'
            status.update(status='complete',completed_at=timed.utc(),results=final['results'])
            write_json(out/'status.json',status)
        except BaseException as error:
            status.update(status='cancelled' if timed.CANCELLED else 'failed',error=repr(error),updated_at=timed.utc())
            write_json(out/'status.json',status)
            raise


if __name__=='__main__':
    def cancel(signum,frame):
        timed.CANCELLED=True
    signal.signal(signal.SIGTERM,cancel)
    signal.signal(signal.SIGINT,cancel)
    main()
