"""Pinned math-only process-error benchmarks; no inferred error categories."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import time
from collections import Counter
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'dataset/prm_math'
PRO_REV = '03b46dab29c1a5117af6742e4c15598b88f79e7f'
SOC_REV = '730dbfa75c6d9cbc6e0117882d22ced43417d86f'
SOC_REPO = 'Xiang-Li-oss/Socratic-PRMBench'

def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')

def info(path):
    return {'path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

def fetch(url):
    for attempt in range(3):
        try:
            return urllib.request.urlopen(url, timeout=60).read()
        except Exception:
            if attempt == 2: raise
            time.sleep(attempt + 1)

def hf_file(repo, revision, filename, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(hf_hub_download(repo, filename, revision=revision, repo_type='dataset'), target)
    return info(target)

def write_dataset(folder, rows, meta):
    path = folder / 'test.parquet'
    pq.write_table(pa.Table.from_pylist(rows), path)
    meta['files'] = {'test': dict(rows=len(rows), **info(path))}
    text_rows = [r for r in rows if not r['image_paths']]
    path = folder / 'test_text_only.parquet'
    pq.write_table(pa.Table.from_pylist(text_rows), path)
    meta['files']['test_text_only'] = dict(rows=len(text_rows), **info(path))
    meta['annotated_error_steps'] = sum(s['is_error'] is True for r in rows for s in r['steps'])
    meta['error_type_counts'] = dict(Counter(s['error_type'] for r in rows for s in r['steps'] if s['error_type']))
    dump(folder / 'manifest.json', meta)
    print(folder.name, len(rows), 'cases;',len(text_rows),'text-only;',meta['annotated_error_steps'],'typed error steps', flush=True)
    return meta

def pro():
    folder = OUT / 'projudgebench_math'
    raw = folder / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    repo = 'julyai/ProJudgeBench'
    sources = [hf_file(repo, PRO_REV, f, raw/f) for f in ['README.md', 'test.parquet']]
    original = pq.read_table(raw/'test.parquet').to_pylist()
    selected = [r for r in original if r['subject'] == 'Math' and r['data_type'] == 'pure_text' and not any(r['image_paths'].values())]
    images = sorted({str(Path(p)) for r in selected for p in r['image_paths'].values() if p})
    def download_image(p):
        assert not Path(p).is_absolute() and '..' not in Path(p).parts
        return hf_file(repo, PRO_REV, p, folder/p)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        image_files = list(pool.map(download_image, images))
    rows = []
    for r in selected:
        steps = []
        for i, a in enumerate(r['process_evaluation']):
            assert len(a) == 4 and a[1] in ['0','1']
            assert a[1] != '0' or a[2].strip()
            steps.append(dict(index=i+1, text=a[0], is_error=a[1]=='0', error_type=a[2] or None,
                              explanation=a[3] or None, label_scope='individual_annotated_step'))
        rows.append(dict(id=r['id'], problem=r['question'], subject='Math', steps=steps,
            image_paths=[str(Path(p)) for p in r['image_paths'].values() if p],
            source_record_json=json.dumps(r, ensure_ascii=False)))
    return write_dataset(folder, rows, dict(name=folder.name, repo=repo, revision=PRO_REV,
        format='process_error_math', source_rows=len(original), selection='subject == Math AND data_type == pure_text AND no image paths',
        raw_files=sources, image_files=image_files, split_usage='evaluation only',
        step_index='1-based position in process_evaluation, not student_solution',
        label_scope='Each process_evaluation entry explicitly supplies text, correctness, type, explanation.'))

def soc():
    folder = OUT/'socratic_prmbench_math'
    raw = folder/'raw'
    raw.mkdir(parents=True, exist_ok=True)
    tree = json.loads(fetch(f'https://api.github.com/repos/{SOC_REPO}/git/trees/{SOC_REV}?recursive=1'))
    files = sorted(x['path'] for x in tree['tree'] if x['path'].endswith('.jsonl'))
    def download(f):
        target=raw/f
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(fetch(f'https://raw.githubusercontent.com/{SOC_REPO}/{SOC_REV}/{f}'))
        return info(target)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        source_files = list(pool.map(download, ['README.md']+files))
    rows, rejected = [], []
    source_count = 0
    for f in files:
        for index, line in enumerate((raw/f).read_text().splitlines()):
            if not line.strip(): continue
            r=json.loads(line)
            source_count += 1
            source = r['idx'].split('#')[0]
            errors = r['error_steps']
            reason = None
            if source not in ['mathbench-A','OlympiadBench','Omni-MATH','gsm8k_test']:
                reason = 'Unverified mathematical source'
            elif not errors or any(not isinstance(k,int) or not 1<=k<=len(r['modified_process']) for k in errors):
                reason = 'Missing or out-of-range 1-based error indices'
            elif not r.get('classification'):
                reason = 'Missing error classification'
            if reason:
                rejected.append(dict(source_file=f, source_index=index, reason=reason, record=r)); continue
            steps=[dict(index=i+1, text=s, is_error=True if i+1 in errors else None,
                error_type=r['classification'] if i+1 in errors else None,
                explanation=r.get('reason') if i+1 in errors else None,
                label_scope='case_classification_linked_to_error_steps' if i+1 in errors else 'unlabeled')
                for i,s in enumerate(r['modified_process'])]
            rows.append(dict(id=f'{Path(f).stem}:{index}',problem=r['modified_question'],subject='Math',
                source_id=r['idx'],steps=steps,image_paths=[],source_file=f,source_index=index,
                source_record_json=json.dumps(r,ensure_ascii=False)))
    dump(folder/'rejected.json', rejected)
    return write_dataset(folder,rows,dict(name=folder.name,repo=SOC_REPO,revision=SOC_REV,
        format='process_error_math',source_rows=source_count,rejected_rows=len(rejected),
        raw_files=source_files,image_files=[],split_usage='evaluation only',
        selection='Verified math source and explicit classification + valid error_steps',
        step_index='1-based error_steps into modified_process',
        label_scope='Case-level synthetic classification linked to listed error steps; not independent per-step classifications. Unlisted steps remain unlabeled.'))

if __name__ == '__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    dump(OUT/'manifest.json',[soc()])
