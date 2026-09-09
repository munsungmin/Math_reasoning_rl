"""Build text/math PERL step classification tables with question-grouped splits."""
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'dataset/perl'
SOURCES={
 'negatives':'5ef7eafcb58fdc0a4f45211141d21437fb410ea5',
 'positives':'b4fb74fd54f4859a9d9a7c10ba63f9a5aba7b723',
 'positives_perturbed':'27ea11d2764f9ddf1467427a154fe93415e6276f'}
TYPES=['Mathematical_Error','Logical_Inconsistency','Accumulation_Error']

def dump(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def file_info(path):return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
def group(question):return hashlib.sha256(re.sub(r'\s+',' ',question).strip().encode()).hexdigest()
def split(g):
    bucket=int(hashlib.sha256(('perl-v1:'+g).encode()).hexdigest()[:8],16)%100
    return 'train' if bucket<80 else 'validation' if bucket<90 else 'test'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    meta=dict(name='perl',format='step_error_classification',sources=[],files={},
        scope='Text mathematical reasoning',primary_error_types=TYPES,
        split_policy='Custom experimental 80/10/10 hash split by whitespace-normalized question; not official splits.',
        source_split='HF train is preserved as source metadata, not an official train/test guarantee.')
    cases=[];steps=[]
    for subset,rev in SOURCES.items():
        repo='PARC-DATASETS/error-detection-'+subset;raw=OUT/'raw'/subset;raw.mkdir(parents=True,exist_ok=True)
        source=dict(repo=repo,revision=rev,files=[])
        for f in ['README.md','data/train-00000-of-00001.parquet']:
            path=raw/Path(f).name
            shutil.copyfile(hf_hub_download(repo,f,repo_type='dataset',revision=rev),path)
            source['files'].append(file_info(path))
        original=pq.read_table(raw/'train-00000-of-00001.parquet').to_pylist()
        source['rows']=len(original);meta['sources'].append(source)
        for index,r in enumerate(original):
            cid=f'{subset}:{index}';g=group(r['question']);sp=split(g)
            annotations=json.loads(r['error_annotation'])['error_annotations']
            premises=json.loads(r['premise_annotation'])
            labels=annotations['step_level'];numbers=[s['step_number'] for s in labels]
            assert len(numbers)==len(set(numbers))
            assert all(isinstance(n,int) and 1<=n<=len(r['steps']) for n in numbers)
            assert r['data_source'] in ['gsm8k','math','metamathqa','orca_math','orcamath']
            cases.append(dict(id=cid,group_id=g,split=sp,subset=subset,source_index=index,
                source_repo=repo,problem=r['question'],data_source=r['data_source'],
                source_record_json=json.dumps(r,ensure_ascii=False)))
            for label in labels:
                n=label['step_number'];types=[e['error_type'] for e in label['errors']]
                assert isinstance(label['has_error'],bool)
                assert label['has_error']==bool(types), (cid,n,label)
                step_premises=[p for p in premises.get('steps',[]) if p.get('step_number')==n]
                steps.append(dict(id=f'{cid}:{n}',case_id=cid,group_id=g,split=sp,subset=subset,
                    problem=r['question'],preceding_steps=r['steps'][:n-1],step_index=n,
                    step_text=r['steps'][n-1],has_error=label['has_error'],error_types=types,
                    label='Correct' if not types else types[0] if len(types)==1 else 'MULTILABEL',
                    annotation_json=json.dumps(label,ensure_ascii=False),
                    premises_json=json.dumps(step_premises,ensure_ascii=False)))
    def save(name,rows):
        path=OUT/(name+'.parquet');pq.write_table(pa.Table.from_pylist(rows),path)
        meta['files'][name]=dict(rows=len(rows),**file_info(path))
    save('cases',cases);save('steps_all',steps)
    primary=[r for r in steps if len(r['error_types'])==1 and r['label'] in TYPES]
    four=[r for r in steps if r['label']=='Correct' or (len(r['error_types'])==1 and r['label'] in TYPES)]
    save('errors_3class',primary);save('steps_4class',four)
    for name,rows in [('errors_3class',primary),('steps_4class',four)]:
        for sp in ['train','validation','test']:save(name+'_'+sp,[r for r in rows if r['split']==sp])
    classes=defaultdict(lambda:dict(steps=0,cases=set(),questions=set(),splits=Counter()))
    for r in steps:
        for label in r['error_types'] or ['Correct']:
            s=classes[label];s['steps']+=1;s['cases'].add(r['case_id']);s['questions'].add(r['group_id']);s['splits'][r['split']]+=1
    meta['class_statistics']={k:dict(steps=v['steps'],cases=len(v['cases']),unique_questions=len(v['questions']),splits=dict(v['splits'])) for k,v in classes.items()}
    meta['unique_questions']=len({r['group_id'] for r in cases})
    meta['chain_level_error_counts']=dict(Counter(e['error_type'] for r in cases for e in json.loads(json.loads(r['source_record_json'])['error_annotation'])['error_annotations'].get('chain_level',{}).get('errors',[])))
    dump(OUT/'manifest.json',meta)
    print(json.dumps({k:meta[k] for k in ['unique_questions','class_statistics','chain_level_error_counts']},indent=2))

if __name__=='__main__':main()
