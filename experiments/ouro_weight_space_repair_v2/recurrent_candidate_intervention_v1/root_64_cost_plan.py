"""CPU-only exact 64-question extension work count; never writes a model plan."""
from pathlib import Path
import hashlib
import itertools
import json
import sys

from tokenizers import Tokenizer

P=Path(__file__).resolve().parent
ROOT=P.parent
H=lambda b:hashlib.sha256(b).hexdigest()
D=lambda v:H(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
read=lambda n:json.loads((P/n).read_text(encoding='utf8'))
selection=read('ROOT_SHA_COHORT.json')
plan=read('INFERENCE_TASK_PLAN.json')
tokenpath=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'artifacts/r24_TOKENIZER_AUDIT_PRIVATE.json'
assert H(tokenpath.read_bytes())=='fcb808fe5e7642f5299be28aea07fc7f6d4f4364c3ac5e408e15a772cbc8fa8d'
tok=Tokenizer.from_file(str(tokenpath))
data={}
for split in ('discovery','selection'):
    data.update((q['id'],q) for q in (json.loads(x) for x in (ROOT/f'data/slcd_{split}.jsonl').read_text(encoding='utf8').splitlines()))
demos=json.loads((ROOT/'data/slcd_fixed_demonstrations.json').read_text(encoding='utf8'))

def question(q,choices):
    return 'Question: '+q+'\n'+'\n'.join(chr(65+i)+'. '+c for i,c in enumerate(choices))+'\nAnswer:'

rendered=[]
for d in demos['documents']:
    ix=d['choices']['label'].index(d['answerKey'])
    rendered.append(question(d['question'],d['choices']['text'])+' '+chr(65+ix))
prefix='\n\n'.join(rendered)+'\n\n'
tasks={t['task_key']:(len(t['input_ids']),len(t['positions'])) for t in plan['tasks']}
original_n=len(tasks)
for qid in selection['first64_ids'][24:]:
    q=data[qid];n=len(q['choices']);gold=q['gold'];identity=list(range(n))
    j=min((i for i in identity if i!=gold),key=lambda i:(H(('2021|'+qid+'|swap|'+str(i)).encode()),i))
    cycle=[(i+1)%n for i in identity];swap=identity.copy();swap[gold],swap[j]=swap[j],swap[gold]
    cyclics={tuple((i+s)%n for i in identity) for s in identity}
    other=next(list(p) for p in itertools.permutations(identity) if p not in cyclics and list(p)!=swap)
    cases=[(q['question'],[q['choices'][i] for i in order]) for order in (identity,cycle,swap,other)]
    modified=list(q['choices']);modified[j]='Not a valid answer'
    cases.extend([(q['question'],modified),('N/A',q['choices']),('N/A',modified)])
    assert len(cases)==7
    for text,choices in cases:
        context=prefix+question(text,choices)
        ids=tok.encode(context,add_special_tokens=False).ids
        for i in range(n):
            combined=tok.encode(context+' '+chr(65+i),add_special_tokens=False).ids
            assert combined[:len(ids)]==ids and len(combined)==len(ids)+1
        pos=[len(ids)-1]
        key=D(dict(ids=ids,positions=pos,depth=4,head_shape=[1,1,2048],primitive='original_uncached_no_explicit_mask_BF16_native_depth4',model_revision='574fa66cb8bf5abdc979642d01cf2b79b16bfab1'))
        tasks[key]=(len(ids),1)
times=json.loads((ROOT/'recurrent_decision_audit_v1/a4_TIMINGS.json').read_text(encoding='utf8'))
warm=[x for x in times if x['purpose']=='parity_native4'][1:]
rate=max(x['seconds']/x['length']**2 for x in warm)
squared=sum(n*n for n,k in tasks.values())
extra=sum(max(0,k-1) for n,k in tasks.values())
forecast=1.5*rate*squared+120+20+.02*4*extra
assert len(tasks)==498 and original_n==218 and forecast>710.6733033489436
result=dict(status='CPU_only_exact_work_estimate_64_not_executed',questions=64,
    primary_conditions_per_question=7,bounded_secondary_B5_B6_same_first4=True,
    unique_native4_body_calls=len(tasks),native_head_calls=4*len(tasks),
    native_head_prediction_positions=4*sum(k for n,k in tasks.values()),
    squared_input_token_sum=squared,added_primary_body_calls=len(tasks)-original_n,
    estimated_GPU_seconds_same_predeclared_policy=forecast,
    remaining_authorized_GPU_seconds_at_preregistration=710.6733033489436,
    estimate_exceeds_balance=True,
    policy='A7warm max(seconds/L^2)*1.5*all L^2 + startup60 + final60 + head reserve20+0.02*extra head positions',
    intervention_results_used=False,model_or_GPU_used=False,
    input_task_plan24_sha256=H((P/'INFERENCE_TASK_PLAN.json').read_bytes()),
    limitation='Conservative planning forecast, not a physical lower bound or measured completion time. No extra 64-question model task is authorized.')
(P/'ROOT_64_COST_PLAN.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print(json.dumps(result,indent=2))
