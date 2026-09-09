"""Prepare MATH levels 3–5 and archive the official IneqMath release (CPU only)."""
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'dataset'


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def file_info(path):
    return dict(path=str(path.relative_to(ROOT)),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write_rows(folder, split, rows):
    path = folder / f'{split}.parquet'
    pq.write_table(pa.Table.from_pylist(rows), path)
    print(folder.name, split, len(rows), flush=True)
    return dict(rows=len(rows), **file_info(path))


def prepare_math():
    parent = json.loads((OUT / 'math/manifest.json').read_text())
    folder = OUT / 'math_level3_5'
    folder.mkdir(exist_ok=True)
    meta = dict(name=folder.name, repo=parent['repo'], url=parent['url'],
                revision=parent['revision'], license=parent['license'],
                format='verl', levels=[3, 4, 5], splits={}, parent_files={},
                note='MATH difficulty levels, for Qwen training; not Qwen-generated data. '
                     'Original train/test split and prior cleaning preserved. '
                     'Per-level files partition each combined split; do not concatenate them with the combined file.')
    for split in ['train', 'test']:
        path = OUT / 'math' / f'{split}.parquet'
        assert file_info(path)['sha256'] == parent['splits'][split]['sha256']
        raw_path = OUT / 'math/raw' / f'{split}-00000-of-00001.parquet'
        raw = pq.read_table(raw_path).to_pylist()
        rows = []
        for row in pq.read_table(path).to_pylist():
            source = raw[row['extra_info']['index']]
            assert source['problem'] == row['extra_info']['question']
            if source['level'] not in ['Level 3', 'Level 4', 'Level 5']:
                continue
            row['extra_info'].update(dataset=folder.name, level=int(source['level'][-1]),
                                     subject=source['type'])
            rows.append(row)
        meta['parent_files'][split] = file_info(path)
        meta['parent_files'][split + '_raw'] = file_info(raw_path)
        meta['splits'][split] = write_rows(folder, split, rows)
        for level in [3, 4, 5]:
            key = f'{split}_level{level}'
            meta['splits'][key] = write_rows(folder, key,
                [r for r in rows if r['extra_info']['level'] == level])
    save_json(folder / 'manifest.json', meta)
    return meta


def prepare_ineq():
    repo = 'AI4Math/IneqMath'
    folder = OUT / 'ineqmath'
    raw = folder / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    previous = folder / 'manifest.json'
    revision = json.loads(previous.read_text())['revision'] if previous.exists() else None
    info = HfApi().dataset_info(repo, revision=revision)
    meta = dict(name='ineqmath', repo=repo, url=f'https://huggingface.co/datasets/{repo}',
                revision=info.sha, license='cc-by-sa-4.0', format='inequality_archive',
                splits={}, raw_files={},
                note='Original answers, solutions and theorem annotations preserved. '
                     'Not directly compatible with the default verl reward dispatcher. '
                     'Test answers are withheld; use dev for local labeled evaluation. '
                     'train_expanded overlaps train; use one or the other. '
                     'Source terms prohibit using test as training data.')
    for filename in ['README.md', 'datasheet.md'] + [f'json/{s}.json' for s in
            ['train', 'train_expanded', 'dev', 'test', 'theorems']]:
        cached = hf_hub_download(repo, filename, repo_type='dataset', revision=info.sha)
        target = raw / Path(filename).name
        shutil.copyfile(cached, target)
        meta['raw_files'][filename] = file_info(target)
    for split in ['train', 'train_expanded', 'dev', 'test']:
        source = json.loads((raw / f'{split}.json').read_text())
        rows = []
        for index, item in enumerate(source):
            assert item['problem'].strip()
            rows.append(dict(data_id=str(item['data_id']), source_index=index,
                source_repo=repo, split=split, problem=item['problem'],
                prompt=[dict(role='user', content=item['problem'])],
                answer=item.get('answer') or None, task_type=item['type'],
                source_record_json=json.dumps(item, ensure_ascii=False)))
        meta['splits'][split] = dict(**write_rows(folder, split, rows),
            labeled_rows=sum(r['answer'] is not None for r in rows),
            task_counts=dict(Counter(r['task_type'] for r in rows)))
    meta['theorem_count'] = len(json.loads((raw / 'theorems.json').read_text()))
    save_json(folder / 'manifest.json', meta)
    return meta


if __name__ == '__main__':
    additions = [prepare_math(), prepare_ineq()]
    path = OUT / 'manifest.json'
    manifest = json.loads(path.read_text())
    names = {m['name'] for m in additions}
    save_json(path, [m for m in manifest if m['name'] not in names] + additions)
