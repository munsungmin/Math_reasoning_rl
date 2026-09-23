"""Supervise a wall-clock-bounded native GRPO run and evaluate saved models.

Only the training launcher receives GPUs 2,4,5,6. After it exits, inference-only
evaluation uses GPU 3. No SFT, teacher targets, Git commands, or test-based model
selection are used. Send SIGTERM to cancel; a STOP file requests stop + evaluation.
"""

import argparse
import datetime
import hashlib
import json
import math
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

from train_verl import REPO, load_settings, read_rows, validate_checkpoint, write_json

BASE = Path('/data/sungmin/math_reasoning')
PRIOR = BASE / 'artifacts/math_error_transfer/overfit_recovery_20260912/pure_rl'
GRADER = BASE / 'envs/limit-rlvr-grader/bin/python'
CANCELLED = False


def utc(timestamp=None):
    return datetime.datetime.fromtimestamp(timestamp or time.time(), datetime.timezone.utc).isoformat()


def complete_jsonl(path):
    if not path.exists():
        return []
    rows = []
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
    return rows


def collapse_detected(history, minimum_step=30, margin=.20, patience=3):
    """Stop only for sustained large validation degradation, not a flat curve."""
    if len(history) < patience + 1:
        return False
    recent = history[-patience:]
    baseline = history[0]['accuracy']
    return recent[-1]['step'] >= minimum_step and all(r['accuracy'] <= baseline - margin for r in recent)


def prune_checkpoints(root, best_step, newest_to_keep=2):
    """Remove only verified older saves in this run; preserve last and best."""
    marker = root / 'latest_checkpointed_iteration.txt'
    if not marker.exists() or not marker.read_text().strip():
        return []
    latest = int(marker.read_text().strip())
    last = root / 'last'
    if last.is_symlink() and last.resolve().parent != root.resolve():
        raise RuntimeError('Refusing checkpoint retention: last points outside this run')
    saves = []
    for path in root.glob('global_step_*'):
        suffix = path.name.removeprefix('global_step_')
        if not suffix.isdigit() or path.is_symlink() or not path.is_dir():
            continue
        step = int(suffix)
        if step <= latest:
            validate_checkpoint(root, step)
            saves.append(step)
    protected = set(sorted(saves)[-newest_to_keep:]) | {latest, best_step}
    if last.is_symlink():
        protected.add(int(last.resolve().name.removeprefix('global_step_')))
    deleted = []
    for step in sorted(saves):
        if step not in protected:
            shutil.rmtree(root / f'global_step_{step}')
            deleted.append(step)
    return deleted


def split_scores(rows, split_ids, metadata=None):
    by_id = {r['sample_id']: r for r in rows}
    assert len(by_id) == len(rows) == (500 if metadata is not None else 367)
    if metadata is None:
        groups = {**split_ids, 'all_levels_3_5': list(by_id)}
    else:
        meta = {r['sample_id']: r for r in metadata}
        assert len(meta) == 500 and set(meta) == set(by_id)
        groups = {**split_ids, 'all_500': list(by_id),
                  'all_levels_3_5': [i for i in by_id if meta[i]['level'] in (3, 4, 5)]}
        groups.update({f'level_{level}': [i for i in by_id if meta[i]['level'] == level] for level in range(1,6)})
    groups['outside_training_split'] = sorted(set(by_id) - set(split_ids['train']))
    return {name: dict(correct=sum(by_id[i]['correct'] for i in ids), samples=len(ids),
                       accuracy=sum(by_id[i]['correct'] for i in ids) / len(ids),
                       length_limit_count=sum(by_id[i]['hit_length_limit'] for i in ids))
            for name, ids in groups.items()}


def stop_child(process):
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=90)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def run_stage(command, env, log_path, timeout):
    with log_path.open('w') as log:
        child = subprocess.Popen(command, cwd=REPO, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        started = time.monotonic()
        try:
            while child.poll() is None:
                if CANCELLED:
                    raise InterruptedError('Supervisor cancelled')
                if time.monotonic() - started > timeout:
                    raise TimeoutError(f'Stage exceeded {timeout} seconds: {log_path}')
                time.sleep(2)
            if child.returncode:
                raise RuntimeError(f'Stage failed with exit {child.returncode}: {log_path}')
        finally:
            stop_child(child)


def wait_for_free_gpus(indices, timeout=600):
    started = time.monotonic()
    while True:
        raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                                       '--format=csv,noheader,nounits'], text=True)
        used = {int(a): int(b) for a, b in (line.split(',') for line in raw.splitlines())}
        if all(used[i] < 100 for i in indices):
            return
        if CANCELLED or time.monotonic() - started > timeout:
            raise RuntimeError(f'GPU cleanup/busy check failed; no other processes were killed: {used}')
        time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--hours', type=float, default=16)
    parser.add_argument('--data-manifest', type=Path,
                        help='Frozen all-levels split for a continuation, instead of the original 264/30/37 run.')
    parser.add_argument('--initialization-provenance', type=Path)
    args = parser.parse_args()
    if not 0 < args.hours <= 24:
        parser.error('hours must be in (0, 24]')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cfg = load_settings(args.config)
    from omegaconf import OmegaConf

    sources = {k: Path(OmegaConf.to_container(cfg.launcher.dataset.sources, resolve=True)[k])
               for k in ('train', 'validation', 'test')}
    source_rows = {k: read_rows(v) for k, v in sources.items()}
    ids = {k: [r['sample_id'] for r in rs] for k, rs in source_rows.items()}
    train = {r['prompt']: r for r in source_rows['train']}
    expected = dict(cfg.launcher.dataset.expected_counts)
    assert len(train) == expected['train'] and {k:len(v) for k,v in ids.items()} == expected
    initial_adapter = Path(OmegaConf.to_container(cfg.actor_rollout_ref.model, resolve=True)['lora_adapter_path'])
    adapter_hash = hashlib.sha256((initial_adapter / 'adapter_model.safetensors').read_bytes()).hexdigest()
    metadata, initialization = None, None
    score_ids = ids
    if args.data_manifest:
        manifest = json.loads(args.data_manifest.read_text())
        assert manifest['split_ids'] == ids and manifest['counts'] == expected == {'train':400,'validation':50,'test':50}
        for name, checksum in manifest['file_sha256'].items():
            assert hashlib.sha256((args.data_manifest.parent/name).read_bytes()).hexdigest() == checksum
        metadata = manifest['records']
        score_ids = {**ids, **{f'previous_{name}': values for name, values in manifest['original_membership_preserved'].items()}}
        assert args.initialization_provenance is not None
        initialization = json.loads(args.initialization_provenance.read_text())
        assert initialization['method'] == 'pure_grpo_continuation'
        assert initialization['adapter_sha256'] == adapter_hash and Path(initialization['adapter']) == initial_adapter
        baseline = None  # User requested training first; no full-benchmark baseline generation here.
        eval_prompts = args.data_manifest.parent / 'evaluation_prompts.jsonl'
    else:
        prior_eval = PRIOR / 'phase_b_step15_math500_level3_5'
        prior_summary = json.loads((prior_eval / 'summary.json').read_text())
        prior_rows = read_rows(prior_eval / 'answers.jsonl')
        by_id = {r['sample_id']: r for r in prior_rows}
        for rows in source_rows.values():
            for row in rows:
                assert row['prompt'] == by_id[row['sample_id']]['input']
                assert row['answer'] == by_id[row['sample_id']]['answer']
        assert adapter_hash == prior_summary['adapter_sha256']
        baseline = split_scores(prior_rows, ids)
        eval_prompts = PRIOR / 'math500_level3_5_data/prompts.jsonl'
    write_json(out / 'baseline.json', baseline)
    shutil.copyfile(eval_prompts, out / 'evaluation_prompts.jsonl')
    started = time.time()
    deadline = started + args.hours * 3600
    env = {**os.environ, 'MATH_RL_DEADLINE': str(deadline), 'WANDB_DISABLE_GIT': 'true',
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false'}
    protocol = dict(method='Pure on-policy GRPO, no SFT', requested_hours=args.hours,
                    started_at=utc(started), deadline=utc(deadline), deadline_epoch=deadline,
                    source_files={k: str(v) for k,v in sources.items()},
                    source_sha256={k: hashlib.sha256(v.read_bytes()).hexdigest() for k,v in sources.items()},
                    split_ids=ids, initial_adapter=str(initial_adapter), initial_adapter_sha256=adapter_hash,
                    config=str(args.config.resolve()), baseline=baseline, initialization=initialization,
                    evaluation_count=500 if metadata is not None else 367,
                    data_manifest=str(args.data_manifest) if args.data_manifest else None,
                    evaluation='Greedy, n=1, 1024 response tokens; test is not used for checkpoint selection',
                    retention='Latest two complete native saves plus validation best; only this run',
                    early_stop='Nonfinite metrics, 30-minute stall, disk under 20 GiB, or three validation scores at least 20 percentage points below initial after step30',
                    stopping='Checkpoint within last five minutes of budget, otherwise interrupt at deadline',
                    final_evaluation='Evaluate last and validation-best (if different) after training exits')
    write_json(out / 'protocol.json', protocol)
    latest_file = Path(OmegaConf.to_container(cfg.launcher, resolve=True)['artifacts_dir']) / f'seed-{cfg.launcher.seed:04d}/latest.json'
    previous = latest_file.read_text() if latest_file.exists() else None
    status = dict(status='initializing', supervisor_pid=os.getpid(), **{k: protocol[k] for k in ('started_at', 'deadline')})
    run = None
    history, progress, visited, seen = [], [], set(), set()
    best_step, best_accuracy = 0, -1.0
    reason = None
    last_activity = time.monotonic()
    with (out / 'launcher.log').open('w') as log:
        child = subprocess.Popen([sys.executable, str(REPO / 'main.py'), '--config-name', args.config.stem],
                                 cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        status['launcher_pid'] = child.pid
        try:
            while child.poll() is None:
                if CANCELLED:
                    raise InterruptedError('Supervisor cancelled')
                if run is None and latest_file.exists() and latest_file.read_text() != previous:
                    run = Path(json.loads(latest_file.read_text())['run'])
                    manifest = json.loads((run / 'manifest.json').read_text())
                    root = Path(manifest['checkpoints'])
                    write_json(out / 'run.json', {'run': str(run), 'checkpoints': str(root)})
                    status.update(run=str(run), checkpoints=str(root), wandb=f"https://wandb.ai/{manifest['wandb_entity']}/math-reasoning/runs/{manifest['wandb_run_id']}")
                if run is not None:
                    for event in complete_jsonl(run / 'metrics.jsonl'):
                        key = (event['session'], event['step'])
                        if key in seen:
                            continue
                        seen.add(key)
                        step, m = event['step'], event['data']
                        last_activity = time.monotonic()
                        if any(isinstance(v, (int,float)) and not math.isfinite(v) for v in m.values()):
                            reason = 'nonfinite_metrics'
                        vk = [k for k in m if k.startswith('val-core/') and '/acc/mean@1' in k]
                        if vk:
                            assert len(vk) == 1
                            score = m[vk[0]]
                            history.append(dict(step=step, accuracy=score))
                            if score > best_accuracy:
                                best_step, best_accuracy = step, score
                            if collapse_detected(history):
                                reason = 'sustained_validation_collapse'
                        if 'train/accuracy' in m:
                            paths = list((run / 'sessions').glob(f'*/rollouts/{step}.jsonl'))
                            assert len(paths) == 1
                            rows = read_rows(paths[0])
                            groups = {}
                            for row in rows:
                                assert row['input'] in train and str(row['gts']) == str(train[row['input']]['answer'])
                                assert row['acc'] in (0,1)
                                groups.setdefault(row['input'], []).append(row)
                                visited.add(train[row['input']]['sample_id'])
                            assert len(groups) == 8 and all(len(rs)==16 for rs in groups.values())
                            assert math.isclose(sum(r['acc'] for r in rows)/128, m['train/accuracy'])
                            progress.append(dict(step=step, accuracy=m['train/accuracy'], trajectories=128,
                                                 seen_train_problems=len(visited), step_seconds=m.get('timing_s/step'),
                                                 entropy=m.get('actor/entropy'), gradient_norm=m.get('actor/grad_norm'),
                                                 length_cap_fraction=m.get('response_length/clip_ratio'),
                                                 all_wrong_groups=sum(all(r['acc']==0 for r in rs) for rs in groups.values())))
                            write_json(out / 'progress.json', progress)
                        if step > 0 and (root / f'global_step_{step}').exists():
                            prune_checkpoints(root, best_step)
                    write_json(out / 'validation.json', history)
                    status.update(status='training', completed_updates=progress[-1]['step'] if progress else 0,
                                  trajectories=sum(r['trajectories'] for r in progress),
                                  unique_training_problems=len(visited), best_validation_step=best_step,
                                  best_validation_accuracy=best_accuracy, latest_validation=history[-1] if history else None,
                                  latest_training=progress[-1] if progress else None)
                    if progress:
                        mean = statistics.mean(r['step_seconds'] for r in progress[-10:])
                        status['estimated_remaining_updates'] = max(0, int((deadline-time.time())/mean))
                    last = root / 'last'
                    if time.time() >= deadline-300 and last.is_symlink():
                        saved_step = int(last.resolve().name.removeprefix('global_step_'))
                        if progress and saved_step == progress[-1]['step']:
                            reason = 'time_budget_checkpoint_boundary'
                if (out / 'STOP').exists():
                    reason = 'user_stop_file'
                if time.time() >= deadline:
                    reason = 'time_budget_deadline'
                if time.monotonic() - last_activity > 1800:
                    reason = 'no_completed_step_for_30_minutes'
                if shutil.disk_usage(out).free < 20*1024**3:
                    reason = 'disk_below_20_gib'
                status.update(updated_at=utc(), remaining_seconds=max(0, deadline-time.time()))
                write_json(out / 'status.json', status)
                if reason:
                    break
                time.sleep(10)
            if reason is None and child.returncode:
                raise RuntimeError(f'Training failed with exit {child.returncode}; see launcher.log')
        finally:
            stop_child(child)
    if run is None or not progress:
        raise RuntimeError('No completed RL update; final evaluation skipped')
    wait_for_free_gpus([2,3,4,5,6])
    last_step = int((root / 'last').resolve().name.removeprefix('global_step_'))
    validate_checkpoint(root, last_step)
    status.update(status='evaluating', stop_reason=reason or 'configured_step_ceiling', training_stopped_at=utc(),
                  saved_last_step=last_step, completed_updates=progress[-1]['step'], actual_training_seconds=time.time()-started)
    write_json(out / 'status.json', status)
    targets = [('last', last_step)]
    if best_step > 0 and best_step != last_step:
        targets.append(('best_validation', best_step))
    elif best_step == 0 and baseline is None:
        targets.append(('best_validation', 0))
    results = {}
    eval_env = {**env, 'CUDA_VISIBLE_DEVICES': '3'}
    model = OmegaConf.to_container(cfg.actor_rollout_ref.model, resolve=True)['path']
    for label, step in targets:
        adapter = initial_adapter
        if step > 0:
            export = out / f'export_step_{step}'
            run_stage([sys.executable, '-m', 'verl.model_merger', 'merge', '--backend', 'fsdp',
                       '--local_dir', str(root / f'global_step_{step}/actor'), '--target_dir', str(export),
                       '--use_cpu_initialization'], {**env, 'CUDA_VISIBLE_DEVICES': ''}, out / f'export_{label}.log', 1800)
            adapter = export / 'lora_adapter'
        evaluation = out / f'evaluation_{label}'
        run_stage([sys.executable, str(REPO / 'script/eval_four_grpo.py'), '--model', model,
                   '--adapter', str(adapter), '--prompts', str(out / 'evaluation_prompts.jsonl'),
                   '--grader-python', str(GRADER), '--out', str(evaluation), '--expected-prompts', str(protocol['evaluation_count']),
                   '--n', '1', '--temperature', '0', '--seed', '17137', '--max-new-tokens', '1024'],
                  eval_env, out / f'evaluation_{label}.log', 7200)
        results[label] = dict(step=step, scores=split_scores(read_rows(evaluation / 'answers.jsonl'), score_ids, metadata),
                             evaluation=str(evaluation), adapter=str(adapter))
    if best_step == 0 and baseline is not None:
        results['best_validation'] = dict(step=0, scores=baseline, adapter=str(initial_adapter),
                                          note='Initial policy retained the highest native validation score; not a newly trained model')
    elif best_step == last_step:
        results['best_validation'] = results['last']
    summary = dict(protocol=protocol, stop=status, baseline=baseline, results=results,
                   selection='Validation only; report last and best without selecting by test score', completed_at=utc())
    write_json(out / 'results.json', summary)
    report = ['# MATH-500 전체 train split 순수 RL 결과', '',
              f'요청 시간: {args.hours:g}시간. 종료 사유: {status["stop_reason"]}. 저장된 마지막 update: {last_step}.', '',
              'SFT 없이 순수 GRPO를 사용했다. Greedy, 문제당 1회, 최대 1024 response tokens.', '',
              '| 평가 집합 | 시작 모델 | 마지막 모델 | Validation으로 선택한 모델 |',
              '|---|---:|---:|---:|']
    for split in results['last']['scores']:
        values = [baseline[split] if baseline else None, results['last']['scores'][split], results['best_validation']['scores'][split]]
        report.append('| '+split+' | '+' | '.join(
            f'{r["correct"]}/{r["samples"]} ({100*r["accuracy"]:.2f}%)' if r is not None else '사전 전체 평가 생략' for r in values)+' |')
    report += ['', f'모델 선택에는 validation만 사용했다. 전체 {protocol["evaluation_count"]}문항 점수에는 이번 train {expected["train"]}문항이 포함된다.',
               '개별 생성·판정은 evaluation_*/answers.jsonl, 설정·시간·출처는 protocol.json과 results.json에 저장했다.']
    (out / 'report_ko.md').write_text('\n'.join(report)+'\n')
    status.update(status='complete', completed_at=utc(), results=str(out/'results.json'))
    write_json(out / 'status.json', status)


if __name__ == '__main__':
    def cancel(signum, frame):
        global CANCELLED
        CANCELLED = True
    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    try:
        main()
    except BaseException as error:
        # Preserve an explicit terminal state, even if initialization or a GPU job fails.
        if '--out' in sys.argv:
            directory = Path(sys.argv[sys.argv.index('--out')+1])
            if directory.exists():
                previous = json.loads((directory/'status.json').read_text()) if (directory/'status.json').exists() else {}
                previous.update(status='cancelled' if CANCELLED else 'failed', error=repr(error), updated_at=utc())
                write_json(directory/'status.json', previous)
        raise
