"""CPU-only fixed-seed, label-free native K1..K4 collection plan."""
from pathlib import Path
import sys,json,hashlib,collections,time
import numpy as np
from tokenizers import Tokenizer
ROOT=Path(__file__).resolve().parent.parent;OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
from r10_core import joint_prefix,joint_requests
from r7_core import digest
from restore_inputs import restore
H=lambda b:hashlib.sha256(b).hexdigest()
def read(p):return json.loads((ROOT/p).read_text(encoding='utf8'))
def lines(p):return [json.loads(x) for x in Path(p).read_text(encoding='utf8').splitlines()]
def main():
 restore();tokpath=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'artifacts/r24_TOKENIZER_AUDIT_PRIVATE.json';assert H(tokpath.read_bytes())=='fcb808fe5e7642f5299be28aea07fc7f6d4f4364c3ac5e408e15a772cbc8fa8d';tok=Tokenizer.from_file(str(tokpath))
 protocol=read('configs/r10_PROTOCOL.json');assert H((ROOT/'data/slcd_fixed_demonstrations.json').read_bytes())==protocol['fixed25_sha256'];prefix=joint_prefix(read('data/slcd_fixed_demonstrations.json'));assert H(prefix.encode())==protocol['joint_demonstrations_sha256'];rows=[]
 for split in ['discovery','selection']:
  assert H((ROOT/('data/slcd_'+split+'.jsonl')).read_bytes())==protocol['dataset_sha256'][split]
  rows+=lines(ROOT/('data/slcd_'+split+'.jsonl'))
 assert len(rows)==576 and len({x['id'] for x in rows})==576
 indices=sorted(np.random.default_rng(2021).choice(576,64,replace=False).tolist());selected=[rows[i] for i in indices];byid={x['id']:x for x in rows}
 hr=json.loads((OUT/'recovered/results/r10_development_initial_COMPLETE.json').read_text(encoding='utf8'));assert hr['status']=='complete' and H((OUT/'recovered/results/r10_development_initial.jsonl').read_bytes())==hr['score_file_sha256'];history={x['id']:x for x in lines(OUT/'recovered/results/r10_development_initial.jsonl')};assert len(history)==576;pr=lines(ROOT/'results/r11_parity_selected.jsonl');parityids=[x['id'] for x in pr];ph={x['id']:x for x in pr}
 def build(row,old):
  n=len(row['choices']);rot=[]
  for shift in range(n):
   order=[(j+shift)%n for j in range(n)];requests=joint_requests(dict(question=row['question'],choices=[row['choices'][i] for i in order]),prefix);pairs=[];memo={};tasks=[]
   for c,a in requests:
    spaces=len(c)-len(c.rstrip())
    if spaces:a=c[-spaces:]+a;c=c[:-spaces]
    pre=tok.encode(c,add_special_tokens=False).ids;tar=tok.encode(c+a,add_special_tokens=False).ids[len(pre):];assert pre and tar
    ids=pre+tar[:-1];positions=list(range(len(pre)-1,len(pre)+len(tar)-1));key=digest(dict(ids=ids,positions=positions))
    if key not in memo:memo[key]=len(tasks);tasks.append(dict(key=key,input_ids=ids,positions=positions))
    pairs.append(dict(task_index=memo[key],target_ids=tar,context_sha256=H(c.encode()),request_sha256=digest((c,a))))
   existing=old['cyclic_rotations'][shift];assert existing['order']==order
   if shift or 'auxiliary_request_sha256' not in old:assert [digest(x) for x in requests]==existing['request_sha256']
   else:assert [digest(x) for x in requests]==old['auxiliary_request_sha256']
   rot.append(dict(shift=shift,order=order,tasks=tasks,pairs=pairs,expected_K1=existing['sum_scores']['native_k1'],expected_K4=existing['sum_scores']['native_k4'],expected_Fixed=existing['sum_scores']['loopcd_k4_fixed']))
  return dict(id=row['id'],nchoices=n,original_choice_char_lengths=row['choice_char_lengths'],rotations=rot)
 parity=[build(byid[x],ph[x]) for x in parityids];cohort=[build(x,history[x['id']]) for x in selected]
 def counts(group):
  tasks=[t for r in group for rot in r['rotations'] for t in rot['tasks']];return dict(questions=len(group),rotations=sum(r['nchoices'] for r in group),unique_depth4_forwards=len(tasks),input_tokens=sum(len(t['input_ids']) for t in tasks),min_input_tokens=min(len(t['input_ids']) for t in tasks),max_input_tokens=max(len(t['input_ids']) for t in tasks),target_token_lengths=sorted({len(p['target_ids']) for r in group for rot in r['rotations'] for p in rot['pairs']}))
 plan=dict(status='CPU_prepared_no_model_authorization',selection=dict(seed=2021,algorithm='np.default_rng(2021).choice(576,64,replace=False), sorted original indices',source_order='discovery384 then selection192',indices=indices,ids=[x['id'] for x in selected],ids_sha256=digest([x['id'] for x in selected])),parity=parity,cohort64=cohort,counts=dict(parity=counts(parity),cohort64=counts(cohort)),historical_development_origin='native1/4 readouts from actual depth5 R10 auxiliary; new depth4 must match endpoints exactly on every rotation/candidate',primitive='original no-KV no-explicit-mask BF16 native normalized states; complete target original FP32 log_softmax/gather/sum; original continuation-shaped head',model_identity=read('configs/r11_PROTOCOL.json')['model_identity'],source_inputs={p:H((ROOT/p).read_bytes()) for p in ['scripts/runtime.py','scripts/ouro_cached_scoring.py','scripts/r7_inference.py','scripts/r10_core.py','scripts/weight_edits.py','configs/r11_PROTOCOL.json','data/slcd_fixed_demonstrations.json','results/r11_parity_selected.jsonl','data/slcd_discovery.jsonl','data/slcd_selection.jsonl','configs/r10_PROTOCOL.json','recurrent_decision_audit_v1/recovered/results/r10_development_initial.jsonl','recurrent_decision_audit_v1/recovered/results/r10_development_initial_COMPLETE.json']})
 plan['counts']['total_model_calls']=plan['counts']['cohort64']['unique_depth4_forwards']+3*plan['counts']['parity']['unique_depth4_forwards'];plan['counts']['depth_weighted_body_calls']=4*(plan['counts']['cohort64']['unique_depth4_forwards']+plan['counts']['parity']['unique_depth4_forwards'])+5*plan['counts']['parity']['unique_depth4_forwards']
 (OUT/'a4_CPU_TASK_PLAN.json').write_text(json.dumps(plan,separators=(',',':'))+'\n',encoding='utf8');(OUT/'a4_CPU_TASK_COUNTS.json').write_text(json.dumps(dict(selection=plan['selection'],counts=plan['counts'],historical_development_origin=plan['historical_development_origin'],plan_sha256=H((OUT/'a4_CPU_TASK_PLAN.json').read_bytes())),indent=2)+'\n',encoding='utf8');print(json.dumps(plan['counts']))
if __name__=='__main__':main()
