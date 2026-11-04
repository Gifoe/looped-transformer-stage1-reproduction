"""CPU tokenizer preparation only; no model loading or GPU calls."""
import json,hashlib,time,os
from pathlib import Path
from r20_admission import ROOT,read,sha,save
from r20_spans import spans

def main():
    os.environ['CUDA_VISIBLE_DEVICES']='';start=time.monotonic()
    assert read('results/r20_CPU_ADMISSION.json')['status']=='admission_pass'
    from transformers import AutoTokenizer
    from runtime import encode_pair
    tok=AutoTokenizer.from_pretrained(str(ROOT/'models/Ouro-1.4B'),local_files_only=True,use_fast=True)
    assert tok.is_fast
    docs=read('data/slcd_fixed_demonstrations.json')['documents']
    demos=[dict(question=d['question'],answer=d['choices']['text'][d['choices']['label'].index(d['answerKey'])]) for d in docs]
    wanted={r['id'] for r in read('configs/r20_COHORT_ADMISSION_LOCK.json')['cohort']};rows={}
    for s in ['discovery','selection']:
        for line in (ROOT/('data/slcd_'+s+'.jsonl')).read_text(encoding='utf8').splitlines():
            r=json.loads(line)
            if r['id'] in wanted:rows[r['id']]=r
    result=[spans(rows[c['id']],tok,encode_pair,demos) for c in read('configs/r20_COHORT_ADMISSION_LOCK.json')['cohort']]
    save('configs/r20_SPANS.json',dict(status='CPU_exact_tokenizer_spans',cohort_sha256=sha('configs/r20_COHORT_ADMISSION_LOCK.json'),rows=result,source_sha256={p:sha('scripts/'+p) for p in ['r20_spans.py','r20_prepare_spans.py']},inputs_sha256={p:sha(p) for p in ['data/slcd_fixed_demonstrations.json','data/slcd_discovery.jsonl','data/slcd_selection.jsonl']},scope='CPU tokenizer only. No load_model, forward, CUDA, labels for spans or heldout.',elapsed_seconds=time.monotonic()-start))
    print('R20_SPANS_COMPLETE',len(result))
if __name__=='__main__':main()
