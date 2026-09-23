"""Apply an explicit final-evaluation request when the evaluator starts.

The already-running timed controller expects a 367-row result. A full 500-row
evaluation is authoritative; a derived 367-row view keeps that controller working
without restarting training or generating the same answers twice.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path


LEVEL_COUNTS = {1: 43, 2: 90, 3: 105, 4: 128, 5: 134}


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def score(rows):
    correct = sum(r['correct'] for r in rows)
    return dict(correct=correct, samples=len(rows), accuracy=correct / len(rows),
                length_limit_count=sum(r['hit_length_limit'] for r in rows),
                extraction_status_counts=dict(Counter(r['extraction_status'] for r in rows)))


def resolve_scope(args):
    """Read the request on process startup, never from the training process."""
    request_path = args.out.parent / 'final_evaluation_request.json'
    if not request_path.exists():
        return None
    request = json.loads(request_path.read_text())
    if args.out.name not in request['evaluation_outputs']:
        return None
    assert request['version'] == 1 and request['scope'] == 'math500_all_levels'
    assert args.expected_prompts == 367 and args.n == 1 and args.temperature == 0
    assert args.max_new_tokens == 1024 and args.seed == 17137
    data = Path(request['data_dir'])
    manifest = json.loads((data / 'manifest.json').read_text())
    prompts_path = data / 'prompts.jsonl'
    assert hashlib.sha256(prompts_path.read_bytes()).hexdigest() == manifest['prompts_sha256']
    assert hashlib.sha256((data / 'manifest.json').read_bytes()).hexdigest() == request['manifest_sha256']
    prompts = read_rows(prompts_path)
    by_id = {r['sample_id']: r for r in prompts}
    meta = {r['sample_id']: r for r in manifest['records']}
    assert len(prompts) == len(by_id) == len(meta) == 500
    assert set(by_id) == set(meta)
    assert Counter(r['level'] for r in meta.values()) == LEVEL_COUNTS
    assert all(set(r) == {'sample_id', 'prompt', 'answer'} for r in prompts)
    previous = read_rows(args.prompts)
    assert len(previous) == len({r['sample_id'] for r in previous}) == 367
    assert {r['sample_id'] for r in previous} == {i for i, r in meta.items() if r['level'] in (3, 4, 5)}
    assert all(r == by_id[r['sample_id']] for r in previous), 'Existing prompts or answers changed'
    splits = request['split_ids']
    assert {k: len(v) for k, v in splits.items()} == {'train': 264, 'validation': 30, 'test': 37}
    split_sets = [set(splits[k]) for k in ('train', 'validation', 'test')]
    assert sum(map(len, split_sets)) == len(set.union(*split_sets)) == 331
    assert all(ids <= set(by_id) for ids in split_sets)
    assert all(meta[i]['level'] in (3, 4, 5) for ids in split_sets for i in ids)
    scope = dict(request=request, request_path=str(request_path),
                 request_sha256=hashlib.sha256(request_path.read_bytes()).hexdigest(),
                 manifest=manifest, previous_ids=[r['sample_id'] for r in previous],
                 compatibility_output=str(args.out), prompts=prompts)
    args.out = args.out.with_name(args.out.name + '_math500_all')
    args.prompts = prompts_path
    args.expected_prompts = 500
    return scope


def publish_results(output, rows, summary, scope):
    """Publish full scores first, then a clearly labelled legacy subset view."""
    by_id = {r['sample_id']: r for r in rows}
    prompts = {r['sample_id']: r for r in scope['prompts']}
    meta = {r['sample_id']: r for r in scope['manifest']['records']}
    assert len(rows) == len(by_id) == 500 and set(by_id) == set(meta)
    assert summary['n_per_problem'] == 1 and summary['temperature'] == 0
    for i, row in by_id.items():
        assert row['input'] == prompts[i]['prompt'] and row['answer'] == prompts[i]['answer']
        assert row['correct'] in (True, False) and row['hit_length_limit'] in (True, False)
    splits = scope['request']['split_ids']
    train = set(splits['train'])
    report = dict(dataset='MATH-500, all 500 questions, levels 1-5',
                  all_500=score(rows),
                  by_level={str(level): score([r for r in rows if meta[r['sample_id']]['level'] == level])
                            for level in LEVEL_COUNTS},
                  splits={k: score([by_id[i] for i in ids]) for k, ids in splits.items()},
                  outside_training_split=score([r for r in rows if r['sample_id'] not in train]),
                  all_levels_3_5=score([by_id[i] for i in scope['previous_ids']]),
                  settings={k: summary[k] for k in ('model', 'adapter', 'adapter_sha256', 'grader_sha256',
                                                   'temperature', 'seed', 'max_new_tokens', 'n_per_problem')},
                  source_sha256=scope['manifest']['source_sha256'],
                  request_sha256=scope['request_sha256'], evaluation=str(output),
                  note='The overall 500 includes the 264 training questions. Outside-training means this split only; validation was used for model selection. No claim about pretraining exposure.')
    write_json(output / 'benchmark_summary.json', report)
    compatibility = Path(scope['compatibility_output'])
    index_path = compatibility.parent / 'math500_all_results.json'
    index = json.loads(index_path.read_text()) if index_path.exists() else dict(
        request=scope['request'], request_sha256=scope['request_sha256'], results={})
    assert index['request_sha256'] == scope['request_sha256']
    index['results'][compatibility.name] = report
    write_json(index_path, index)
    lines = ['# MATH-500 전체 500문항 평가', '',
             '학습 종료 후 greedy, 문제당 1회, 최대 1024 response tokens. 모델 선택은 기존 validation 기준.', '',
             '| 평가 모델 | 전체 500 | Level 1 | Level 2 | Level 3 | Level 4 | Level 5 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, result in index['results'].items():
        values = [result['all_500']] + [result['by_level'][str(level)] for level in LEVEL_COUNTS]
        lines.append('| ' + name + ' | ' + ' | '.join(
            f'{v["correct"]}/{v["samples"]} ({100*v["accuracy"]:.2f}%)' for v in values) + ' |')
    lines += ['', '전체 점수에는 이번 train 264문항이 포함된다. train 제외 236문항과 기존 validation30/test37 점수는 JSON에 따로 저장했다.',
              '개별 생성·채점 및 checkpoint 출처: evaluation_*_math500_all/answers.jsonl, summary.json, benchmark_summary.json.',
              '기존 evaluation_* 폴더는 같은 500회 생성 결과에서 추출한 level 3~5의 367문항 집계이며 추가 추론을 하지 않았다.']
    (compatibility.parent / 'math500_all_report_ko.md').write_text('\n'.join(lines) + '\n')
    subset = [by_id[i] for i in scope['previous_ids']]
    subset_score = score(subset)
    subset_summary = {**summary, **{k: subset_score[k] for k in ('correct', 'samples', 'accuracy', 'extraction_status_counts')},
                      'method': 'Derived level 3-5 subset of a single full MATH-500 evaluation; no additional generation',
                      'full_evaluation': str(output),
                      'length_limit_rate': subset_score['length_limit_count'] / len(subset),
                      'per_problem': {i: summary['per_problem'][i] for i in scope['previous_ids']},
                      'prompt_sha256': {i: summary['prompt_sha256'][i] for i in scope['previous_ids']},
                      'elapsed_seconds_note': 'Elapsed time belongs to the full 500 evaluation.'}
    compatibility.mkdir(exist_ok=False)
    (compatibility / 'answers.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in subset))
    write_json(compatibility / 'summary.json', subset_summary)
