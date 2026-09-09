"""Check RFM source alignment and complete multi-label extraction, without APIs."""
import json,hashlib
from pathlib import Path
import pyarrow.parquet as pq
from prepare_rfm import parse,proof_for_judge,CATEGORIES
ROOT=Path(__file__).resolve().parents[1];folder=ROOT/'dataset/rfm'
m=json.loads((folder/'manifest.json').read_text())
for f in m['raw_files']+list(m['files'].values()):
 p=ROOT/f['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256']
 if 'rows' in f:assert pq.read_table(p).num_rows==f['rows']
proofs=pq.read_table(folder/'proofs.parquet').to_pylist();lookup={r['proof_id']:r for r in proofs}
assert len(lookup)==len(proofs)
raw=folder/'raw';cache={}
def read(p):
 if p not in cache:cache[p]=json.loads(p.read_text(encoding='utf-8-sig'))
 return cache[p]
for r in proofs:
 k=r['knowledge_level'];i=r['source_index']
 assert r['proof']==read(raw/'answers'/f"{r['model']}_all.json")[k][i]
 assert r['problem']==read(raw/'data'/f'{k}_combined.json')[i]['statement']
 assert r['proof_for_judge']==proof_for_judge(r['proof'])
ratings=pq.read_table(folder/'ratings_all.parquet').to_pylist()
for r in ratings:
 p=lookup[r['proof_id']]
 assert r['problem']==p['problem'] and r['proof']==p['proof']
 assert r['proof_for_judge']==p['proof_for_judge']
 assert r['judgement_text']==read(raw/r['source_file'])[r['knowledge_level']][r['source_index']]
 labels,correct=parse(r['judgement_text'])
 assert labels==r['label_vector'] and correct==r['overall_correctness']
 assert r['error_types']==[c for c,v in zip(CATEGORIES,labels) if v]
reference=pq.read_table(folder/'classification_reference.parquet').to_pylist()
assert len({r['proof_id'] for r in reference})==len(reference)
assert reference==[r for r in ratings if r['judge']==m['reference_judge'] and r['label_consistent']]
print('PASS: raw/processed hashes, positional problem-answer-judge alignment, complete labels, reference judge uniqueness')
print(len(proofs),'proofs;',len(ratings),'ratings;',len(reference),'reference classifications')
