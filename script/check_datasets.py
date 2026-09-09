"""Validate prepared files against manifests and check exact train/eval overlap."""
import hashlib
import json
from pathlib import Path
import re
import pyarrow.parquet as pq
root = Path(__file__).resolve().parents[1]
manifest = json.loads((root/'dataset/manifest.json').read_text())
questions = {}
for dataset in manifest:
    for info in list(dataset.get('raw_files', {}).values()) + list(dataset.get('parent_files', {}).values()):
        path = root / info['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == info['sha256'], path
    for split, info in dataset['splits'].items():
        path = root / info['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == info['sha256'], path
        rows = pq.read_table(path).to_pylist()
        assert len(rows) == info['rows'] > 0
        if dataset.get('format') == 'inequality_archive':
            assert sum(r['answer'] is not None for r in rows) == info['labeled_rows']
            for row in rows:
                original = json.loads(row['source_record_json'])
                assert row['problem'] == original['problem'] == row['prompt'][0]['content']
                assert row['answer'] == (original.get('answer') or None)
                assert row['source_repo'] == dataset['repo']
                assert row['task_type'] in ['bound', 'relation']
                assert (row['answer'] is None) == (split == 'test')
            questions[(dataset['name'], split)] = {re.sub(r'\s+', ' ', r['problem']).strip() for r in rows}
            print(dataset['name'], split, len(rows), 'archive/labels/hash PASS')
            continue
        for row in rows:
            assert row['prompt'][0]['role'] == 'user'
            assert row['prompt'][0]['content'].strip()
            assert row['reward_model']['ground_truth'].strip()
            assert row['extra_info']['source_repo'] == dataset['repo']
        questions[(dataset['name'],split)] = {re.sub(r'\s+', ' ', r['extra_info']['question']).strip() for r in rows}
        print(dataset['name'], split, len(rows), 'schema/answers/hash PASS')
for train, test in [(('gsm8k','train'),('gsm8k','test'))] + [(('math','train'),(d,'test')) for d in ['math','math500','aime2024','aime2025']]:
    overlap = questions[train] & questions[test]
    assert not overlap, (train, test, len(overlap))
    print('No exact normalized train/eval overlap:', train, test)
assert len(questions['math','test'] & questions['math500','test']) == 500
if ('math_level3_5', 'train') in questions:
    for split in ['train', 'test']:
        combined = pq.read_table(root / f'dataset/math_level3_5/{split}.parquet').to_pylist()
        source = pq.read_table(root / f'dataset/math/{split}.parquet').to_pylist()
        raw = pq.read_table(root / f'dataset/math/raw/{split}-00000-of-00001.parquet').to_pylist()
        expected = {r['extra_info']['index'] for r in source
                    if raw[r['extra_info']['index']]['level'] in ['Level 3', 'Level 4', 'Level 5']}
        assert {r['extra_info']['index'] for r in combined} == expected
        source_by_index = {r['extra_info']['index']: r for r in source}
        for row in combined:
            index = row['extra_info']['index']
            assert raw[index]['level'] == f"Level {row['extra_info']['level']}"
            assert row['reward_model'] == source_by_index[index]['reward_model']
        total = 0
        for level in [3, 4, 5]:
            part = pq.read_table(root / f'dataset/math_level3_5/{split}_level{level}.parquet').to_pylist()
            assert part == [r for r in combined if r['extra_info']['level'] == level]
            total += len(part)
        assert total == len(combined)
    for d in ['math', 'math500', 'aime2024', 'aime2025']:
        assert not questions['math_level3_5', 'train'] & questions[d, 'test']
    print('MATH level selection / partition / parent answers / evaluation overlap PASS')
if ('ineqmath', 'train') in questions:
    for split in ['train', 'train_expanded']:
        for evaluation in ['dev', 'test']:
            assert not questions['ineqmath', split] & questions['ineqmath', evaluation], (split, evaluation)
    assert not questions['ineqmath', 'dev'] & questions['ineqmath', 'test']
    print('IneqMath exact normalized train/dev/test overlap PASS')
print('ALL DATASET CHECKS PASSED (MATH-500 is contained in MATH test)')
