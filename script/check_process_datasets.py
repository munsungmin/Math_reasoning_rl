"""Validate process labels, raw provenance, and question-grouped classification splits."""
import hashlib,json
from pathlib import Path
from collections import Counter
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1]

def verify(info):
    p=ROOT/info['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==info['sha256'],p
    if 'rows' in info:assert pq.read_table(p).num_rows==info['rows']

m=json.loads((ROOT/'dataset/perl/manifest.json').read_text())
for s in m['sources']:
    for f in s['files']:verify(f)
for f in m['files'].values():verify(f)
cases={r['id']:r for r in pq.read_table(ROOT/'dataset/perl/cases.parquet').to_pylist()}
steps=pq.read_table(ROOT/'dataset/perl/steps_all.parquet').to_pylist()
assert len({r['id'] for r in steps})==len(steps)
for r in steps:
    raw=json.loads(cases[r['case_id']]['source_record_json']);n=r['step_index']
    assert r['step_text']==raw['steps'][n-1]
    assert r['preceding_steps']==raw['steps'][:n-1]
    a=next(s for s in json.loads(raw['error_annotation'])['error_annotations']['step_level'] if s['step_number']==n)
    assert r['error_types']==[e['error_type'] for e in a['errors']]
    assert r['has_error']==a['has_error']
for name in ['errors_3class','steps_4class']:
    all_rows=pq.read_table(ROOT/f'dataset/perl/{name}.parquet').to_pylist()
    groups={};ids=set()
    for sp in ['train','validation','test']:
        rows=pq.read_table(ROOT/f'dataset/perl/{name}_{sp}.parquet').to_pylist()
        assert all(r['split']==sp for r in rows)
        groups[sp]={r['group_id'] for r in rows}
        assert not ids & {r['id'] for r in rows}
        ids.update(r['id'] for r in rows)
    assert ids=={r['id'] for r in all_rows}
    assert not groups['train'] & groups['test']
    assert not groups['train'] & groups['validation']
    assert not groups['validation'] & groups['test']
    print(name,len(all_rows),dict(Counter(r['label'] for r in all_rows)),'source labels and disjoint question groups PASS')
for m in json.loads((ROOT/'dataset/prm_math/manifest.json').read_text()):
    for f in m['raw_files']:verify(f)
    for f in m['files'].values():verify(f)
    rows=pq.read_table(ROOT/m['files']['test']['path']).to_pylist()
    for r in rows:
        raw=json.loads(r['source_record_json'])
        assert not r['image_paths']
        assert len(r['steps'])==len(raw['modified_process'])
        for s in r['steps']:
            assert s['text']==raw['modified_process'][s['index']-1]
            assert (s['is_error'] is True)==(s['index'] in raw['error_steps'])
            assert s['error_type']==(raw['classification'] if s['index'] in raw['error_steps'] else None)
    assert len(rows)+m['rejected_rows']==m['source_rows']
    print(m['name'],len(rows),'indices/classification/raw hashes PASS')
print('ALL PROCESS DATASET CHECKS PASSED')
