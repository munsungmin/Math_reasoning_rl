"""Archive RFM proofs and parse explicit proof-level multi-label judge ratings."""
import concurrent.futures
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
import urllib.request
import time
import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'dataset/rfm'
REV='d73be090eb811e61d509e83e29b9b29bf5edb84f'
REPO='guodadi/RFMDataset'
CATEGORIES=['Transformation Error','Over Generalization','Invalid Construction','Wrong Division',
 'Circular Reasoning','Logic Violation','Hidden Assumption','Boundary Neglect','Vague Argument','Incomplete Proof','Others']
REFERENCE_JUDGE='gemini-2.5-pro-preview-0506'

def fetch(url):
    for attempt in range(3):
        try:return urllib.request.urlopen(url,timeout=60).read()
        except Exception:
            if attempt==2:raise
            time.sleep(attempt+1)

def dump(path,x):path.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def info(p):return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def parse(text):
    # Never turn a missing label into a negative label.
    clean=text.replace('**','')
    if '### Error Pattern Analysis' not in clean or '### Overall Correctness' not in clean:return None
    block=clean.rsplit('### Error Pattern Analysis',1)[1]
    errors,overall=block.split('### Overall Correctness',1)
    values=[]
    for c in CATEGORIES:
        matches=re.findall(r'^\s*-?\s*'+re.escape(c)+r'\s*:\s*([01])\b',errors,re.M|re.I)
        if len(matches)!=1:return None
        values.append(int(matches[0]))
    m=re.match(r'\s*[:\-]?\s*([01])\b',overall)
    if not m:return None
    return values,int(m.group(1))

def proof_for_judge(text):
    # Match upstream strip_thinking; retain the original answer separately.
    if '</think>' not in text:return text
    thought,proof=text.split('</think>',1)
    proof=proof.strip()
    return thought.removeprefix('<think>').strip() if len(proof.split())<10 else proof

def main():
    raw=OUT/'raw';raw.mkdir(parents=True,exist_ok=True)
    tree=json.loads(fetch(f'https://api.github.com/repos/{REPO}/git/trees/{REV}?recursive=1'))
    names=[x['path'] for x in tree['tree'] if x['type']=='blob' and
        (x['path'].startswith(('data/','answers/','judgements/')) or x['path'] in
         ['README.md','LICENSE','prompt/eval_prompt.txt','rfmdataset/data.py','rfmdataset/evaluation.py'])]
    def download(name):
        p=raw/name;p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists():p.write_bytes(fetch(f'https://raw.githubusercontent.com/{REPO}/{REV}/{name}'))
        return info(p)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:raw_files=list(pool.map(download,names))
    problems={k:read(raw/f'data/{k}_combined.json') for k in ['ms','hs','ug']}
    assert {k:len(v) for k,v in problems.items()}=={'ms':52,'hs':88,'ug':60}
    answers={p.name.removesuffix('_all.json'):read(p) for p in (raw/'answers').glob('*_all.json')}
    proofs=[];lookup={};missing_proofs=[]
    for model,levels in sorted(answers.items()):
        for level,ps in problems.items():
            assert len(levels[level])==len(ps),(model,level)
            for i,(p,text) in enumerate(zip(ps,levels[level])):
                pid=f'{level}:{p["id"]}';uid=f'{model}:{pid}'
                row=dict(proof_id=uid,problem_id=pid,model=model,knowledge_level=level,source_index=i,
                    problem=p['statement'],proof=text,proof_for_judge=proof_for_judge(text),problem_metadata_json=json.dumps(p,ensure_ascii=False))
                if not isinstance(text,str) or not text.strip():
                    missing_proofs.append(dict(model=model,level=level,index=i,reason='Empty or non-string answer'));continue
                proofs.append(row);lookup[(model,level,i)]=row
    ratings=[];rejected=[];skipped=[]
    for p in sorted((raw/'judgements').glob('*.json')):
        if not p.name.endswith('_all.json'):
            skipped.append(dict(file=p.name,reason='App/reflection variant; alignment to original published answers not assumed.'));continue
        matching=[m for m in answers if p.name.startswith(m+'_')]
        if len(matching)!=1:
            skipped.append(dict(file=p.name,reason='No unique matching answer file'));continue
        model=matching[0];judge=p.name[len(model)+1:-len('_all.json')];data=read(p)
        for level,ps in problems.items():
            values=data.get(level,[])
            if len(values)!=len(ps):
                skipped.append(dict(file=p.name,level=level,reason='Length mismatch',rows=len(values)));continue
            for i,text in enumerate(values):
                if (model,level,i) not in lookup:
                    rejected.append(dict(file=p.name,level=level,index=i,reason='Missing proof text'));continue
                parsed=parse(text) if isinstance(text,str) else None
                if parsed is None:
                    rejected.append(dict(file=p.name,level=level,index=i,reason='Missing or ambiguous explicit labels'));continue
                labels,correct=parsed;proof=lookup[(model,level,i)]
                consistent=correct==int(not any(labels))
                ratings.append(dict(rating_id=f'{p.stem}:{level}:{i}',proof_id=proof['proof_id'],
                    problem_id=proof['problem_id'],model=model,judge=judge,source_file='judgements/'+p.name,
                    knowledge_level=level,source_index=i,problem=proof['problem'],proof=proof['proof'],proof_for_judge=proof['proof_for_judge'],
                    label_scope='whole_proof',label_source='LLM_as_judge',label_vector=labels,
                    error_types=[c for c,v in zip(CATEGORIES,labels) if v],overall_correctness=correct,
                    label_consistent=consistent,judgement_text=text))
    reference=[r for r in ratings if r['judge']==REFERENCE_JUDGE and r['label_consistent']]
    assert len({r['proof_id'] for r in reference})==len(reference)
    meta=dict(name='rfm',repo=REPO,revision=REV,format='whole_proof_multilabel',
        categories=CATEGORIES,raw_files=raw_files,files={},unique_problems=200,
        reference_judge=REFERENCE_JUDGE,parsed_ratings=len(ratings),rejected_ratings=len(rejected),
        inconsistent_ratings=sum(not r['label_consistent'] for r in ratings),
        skipped_files=skipped,missing_proofs=missing_proofs,proofs=len(proofs),models=len(answers),
        note='Published LLM judge labels over whole proofs. Ratings by different judges are not independent proofs. No step labels or train/test split invented.')
    for name,rows in [('proofs',proofs),('ratings_all',ratings),('classification_reference',reference)]:
        p=OUT/f'{name}.parquet';pq.write_table(pa.Table.from_pylist(rows),p)
        meta['files'][name]=dict(rows=len(rows),**info(p))
    meta['reference_type_counts']={c:sum(c in r['error_types'] for r in reference) for c in CATEGORIES}
    meta['reference_unique_questions']=len({r['problem_id'] for r in reference})
    meta['reference_error_count_distribution']=dict(Counter(len(r['error_types']) for r in reference))
    meta['judges']=dict(Counter(r['judge'] for r in ratings))
    dump(OUT/'rejected.json',rejected);dump(OUT/'manifest.json',meta)
    dump(OUT/'example_multilabel.json',next(r for r in reference if len(r['error_types'])>=2))
    print(json.dumps({k:meta[k] for k in ['proofs','models','parsed_ratings','rejected_ratings','inconsistent_ratings','reference_unique_questions','reference_type_counts','reference_error_count_distribution','judges']},indent=2))

if __name__=='__main__':main()
