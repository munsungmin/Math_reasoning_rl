"""Download pinned raw data and build verl parquet without importing torch/CUDA."""
import hashlib
import importlib.util
import json
import re
from pathlib import Path
import shutil
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'dataset'
# Load only the standalone answer parser, not verl's CUDA-importing package init.
spec = importlib.util.spec_from_file_location('math_score', ROOT / 'verl-cu121/verl/utils/reward_score/math.py')
math_score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(math_score)
SPECS = [
 ('gsm8k', 'openai/gsm8k', 'main', [('train','main/train-00000-of-00001.parquet'),('test','main/test-00000-of-00001.parquet')]),
 ('math', 'DigitalLearningGmbH/MATH-lighteval', 'default', [('train','data/train-00000-of-00001.parquet'),('test','data/test-00000-of-00001.parquet')]),
 ('math500', 'HuggingFaceH4/MATH-500', 'default', [('test','test.jsonl')]),
 ('aime2025', 'opencompass/AIME2025', 'AIME2025-I+II', [('test_I','aime2025-I.jsonl'),('test_II','aime2025-II.jsonl')]),
 ('aime2024', 'HuggingFaceH4/aime_2024', 'default', [('test','data/train-00000-of-00001.parquet')]),
]
manifest = []
for name, repo, config, splits in SPECS:
    folder = OUT / name
    raw = folder / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    meta_path = folder / 'manifest.json'
    previous = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    info = HfApi().dataset_info(repo, revision=previous.get('revision'))
    revision = info.sha
    record = dict(name=name, repo=repo, url=f'https://huggingface.co/datasets/{repo}',
                  config=config, revision=revision, license=(info.card_data or {}).get('license'), splits={})
    readme = hf_hub_download(repo, 'README.md', repo_type='dataset', revision=revision)
    shutil.copy2(readme, raw / 'SOURCE_README.md')
    for split, file in splits:
        cached = hf_hub_download(repo, file, repo_type='dataset', revision=revision)
        target = raw / Path(file).name
        shutil.copy2(cached, target)
        source = pq.read_table(target).to_pylist() if file.endswith('.parquet') else [json.loads(l) for l in target.read_text().splitlines() if l.strip()]
        rows = []
        rejected = []
        for i, ex in enumerate(source):
            question = ex.get('question', ex.get('problem'))
            if name == 'gsm8k':
                answer = ex['answer'].rsplit('####', 1)[1].strip().replace(',', '')
                instruction = 'Let\'s think step by step and output the final answer after "####".'
                score_source = 'openai/gsm8k'
            else:
                answer = ex.get('answer')
                if answer is None:
                    answer = math_score.remove_boxed(math_score.last_boxed_only_string(ex['solution']))
                instruction = r"Let's think step by step and output the final answer within \boxed{}."
                # This verl version dispatches the symbolic math scorer by data_source.
                # Keep the real dataset identity separately in extra_info and manifest.
                score_source = 'DigitalLearningGmbH/MATH-lighteval'
            assert isinstance(question, str) and question.strip(), (name, i, ex)
            if answer is None or not str(answer).strip():
                rejected.append(dict(index=i, reason='missing_or_empty_final_answer', original=ex))
                continue
            rows.append(dict(data_source=score_source,
                prompt=[dict(role='user', content=question+' '+instruction)], ability='math',
                reward_model=dict(style='rule', ground_truth=str(answer)),
                extra_info=dict(split=split, index=i, dataset=name, source_repo=repo,
                    question=question, solution=str(ex.get('solution', ex.get('answer',''))))))
        output = folder / f'{split}.parquet'
        pq.write_table(pa.Table.from_pylist(rows), output)
        assert pq.read_table(output).num_rows + len(rejected) == len(source)
        (folder / f'{split}.rejected.json').write_text(json.dumps(rejected, indent=2, ensure_ascii=False)+'\n')
        record['splits'][split] = dict(rows=len(rows), source_rows=len(source), rejected=len(rejected), source_file=file, path=str(output.relative_to(ROOT)),
            sha256=hashlib.sha256(output.read_bytes()).hexdigest())
        print(name, split, len(rows), output, flush=True)
    if name == 'aime2025':
        combined = pa.concat_tables([pq.read_table(folder / f'{part}.parquet') for part in ['test_I','test_II']])
        pq.write_table(combined, folder / 'test.parquet')
        record['splits']['test'] = dict(rows=combined.num_rows, path=str((folder/'test.parquet').relative_to(ROOT)), source_splits=['test_I','test_II'], sha256=hashlib.sha256((folder/'test.parquet').read_bytes()).hexdigest())
        print(name, 'test combined', combined.num_rows, flush=True)
    if name == 'aime2024':
        record['note'] = 'Upstream split is named train; stored as test locally because this is an evaluation benchmark.'
    if name in ('math500','aime2024','aime2025'):
        record['reward_dispatch'] = 'DigitalLearningGmbH/MATH-lighteval selects the existing symbolic scorer; true source is preserved in extra_info.source_repo.'
    meta_path.write_text(json.dumps(record, indent=2, ensure_ascii=False)+'\n')
    manifest.append(record)
# Exclude exact whitespace-normalized evaluation questions from MATH training.
normalize = lambda text: re.sub(r"\s+", " ", text).strip()
math_folder = OUT / 'math'
train_rows = pq.read_table(math_folder / 'train.parquet').to_pylist()
eval_questions = set()
for rel in ['math/test.parquet', 'math500/test.parquet', 'aime2024/test.parquet', 'aime2025/test.parquet']:
    eval_questions.update(normalize(r['extra_info']['question']) for r in pq.read_table(OUT / rel).to_pylist())
removed = [r for r in train_rows if normalize(r['extra_info']['question']) in eval_questions]
train_rows = [r for r in train_rows if normalize(r['extra_info']['question']) not in eval_questions]
pq.write_table(pa.Table.from_pylist(train_rows), math_folder / 'train.parquet')
(math_folder / 'train.eval_overlap_removed.json').write_text(json.dumps(removed, indent=2, ensure_ascii=False)+'\n')
math_meta = next(m for m in manifest if m['name'] == 'math')
math_meta['splits']['train'].update(rows=len(train_rows), eval_overlap_removed=len(removed), sha256=hashlib.sha256((math_folder/'train.parquet').read_bytes()).hexdigest())
(math_folder / 'manifest.json').write_text(json.dumps(math_meta, indent=2, ensure_ascii=False)+'\n')
print('MATH evaluation overlap removed:', len(removed), 'remaining train:', len(train_rows), flush=True)
# Preserve separately prepared datasets in the shared catalog.
catalog_path = OUT / 'manifest.json'
if catalog_path.exists():
    names = {m['name'] for m in manifest}
    manifest.extend(m for m in json.loads(catalog_path.read_text()) if m['name'] not in names)
catalog_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n')
print('ALL DATASETS PREPARED', flush=True)
