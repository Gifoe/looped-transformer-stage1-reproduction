"""CPU-only, freeze label-independent R20 cohort before baseline admission.
No tokenizer/model/CUDA imports; no heldout data; immutable exclusive files.
"""
import json, hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def read(p):return json.loads((ROOT/p).read_text(encoding='utf8'))
def save(p,x):
    assert Path(p).name.startswith('r20_')
    with (ROOT/p).open('x',encoding='utf8') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def main():
    source='configs/r16_EXACT_KL_PILOT_LOCK.json'
    if not (ROOT/source).exists():
        found=list((ROOT/'configs').glob('r16_*LOCK.json'));assert len(found)==1;source=str(found[0].relative_to(ROOT)).replace('\\','/')
    original=read(source)['cohort'];assert len(original)==64
    cohort=[original[int(i)] for i in np.random.default_rng(2021).permutation(64)[:24]]
    protocol=dict(status='immutable_cohort_before_baseline_admission',cohort=cohort,selection='default_rng(2021).permutation(original_R16_64)[:24]; no outcome reads before exclusive cohort write',input_sha256={source:sha(source),'r20_ASTRA_LITERATURE_REVIEW.md':sha('r20_ASTRA_LITERATURE_REVIEW.md')},source_sha256=sha('scripts/r20_admission.py'),scope='CPU-only baseline admission. No new features, tokenizer, model, CUDA, heldout, external dataset or GPU worker. This is not GPU authorization.')
    path='configs/r20_COHORT_ADMISSION_LOCK.json';save(path,protocol)
    # Deliberately only after durable exclusive cohort creation.
    wanted={x['id'] for x in cohort};rows={}
    inputs={}
    for split in ['discovery','selection']:
        p='results/r7_'+split+'_development.jsonl';inputs[p]=sha(p)
        with (ROOT/p).open(encoding='utf8') as f:
            for line in f:
                r=json.loads(line)
                if r['id'] in wanted:rows[r['id']]=r
    assert set(rows)==wanted
    coverage=[]
    for x in cohort:
        r=rows[x['id']];b=r['baseline_scores'];gold=r['gold']
        rc=int(np.argmax(b['M0_R6']))==gold;fc=int(np.argmax(b['fixed']))==gold
        coverage.append(dict(id=x['id'],R6_correct=rc,Fixed_correct=fc,R6_unique=rc and not fc))
    n=sum(x['R6_unique'] for x in coverage)
    out=dict(status='admission_pass' if n else 'admission_NoGo',cohort_lock_sha256=sha(path),n=24,R6_unique_vs_Fixed=n,coverage=coverage,input_sha256=inputs,new_effects_computed=False,GPU_worker_started=False,heldout_accessed=False,automatic_GPU_authorization=False,on_zero='Stop without resampling, increasing cohort or changing gate.')
    save('results/r20_CPU_ADMISSION.json',out);print(json.dumps(out,indent=2))
if __name__=='__main__':main()
