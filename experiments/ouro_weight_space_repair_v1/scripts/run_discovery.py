"""Training-only native recurrence audit and declared readout paths; no test access."""
from pathlib import Path
import argparse,hashlib,json,time
from runtime import ROOT,load_model,encode_pair,save,check_budget,outcome
from ouro_cached_scoring import OuroCachedScorer

def hashfile(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--n',type=int,default=64);ap.add_argument('--parity',type=int,default=3);ap.add_argument('--uncached',action='store_true');ap.add_argument('--study');a=ap.parse_args()
 gate=json.loads((ROOT/'results/READINESS.json').read_text());assert gate['status']=='passed'
 if not a.uncached:assert gate.get('allowed_scoring_path')!='native_uncached_only','Cached optimization not admitted'
 rows=[json.loads(x) for x in (ROOT/'data/discovery.jsonl').read_text().splitlines()][:a.n]
 assert len(rows)==a.n and 1<=a.n<=192
 codehash=hashlib.sha256(''.join(hashfile(ROOT/'scripts'/n) for n in ['runtime.py','ouro_cached_scoring.py','run_discovery.py']).encode()).hexdigest()
 start=time.time();tok,model=load_model();scorer=OuroCachedScorer(model,6)
 import torch,numpy as np
 name=a.study or ('discovery_uncached' if a.uncached else 'discovery_native');assert name.replace('_','').isalnum()
 out=ROOT/'results'/name;out.mkdir(exist_ok=True)
 shared_ids=tok.encode((ROOT/'data/shared_demo_prefix.txt').read_text(),add_special_tokens=False)[:-2]
 pre=time.time();shared=scorer.build_shared_prefix(shared_ids) if not a.uncached else None;prefill_seconds=time.time()-pre
 completed={}
 path=out/'scores.jsonl'
 if path.exists():
  for line in path.read_text().splitlines():
   r=json.loads(line);assert r['implementation_sha256']==codehash and r['cached']==(not a.uncached);assert r['id'] not in completed;completed[r['id']]=r
 parity_path=out/'PARITY.json'
 if not parity_path.exists():
  cases=[]
  for row in rows[:a.parity]:
   check_budget(60);pairs=[encode_pair(tok,*p) for p in row['request_pairs']]
   if a.uncached:r=dict(passed=True,path='native_no_cache')
   else:r=scorer.compare_cached_uncached(pairs,shared=shared,char_lengths=[max(n,1) for n in row['choice_char_lengths']],score_atol=.01,token_atol=.05)
   cases.append(dict(id=row['id'],**r));save(str(out.relative_to(ROOT)/'PARITY_PROGRESS.json'),dict(cases=cases))
   assert r['passed'],('25shot native/cache parity failed; no scientific scores accepted',row['id'],r)
  # lm-eval oracle checks the unchanged scoring/token boundary, without labels.
  from lm_eval.models.huggingface import HFLM
  from lm_eval.api.instance import Instance
  row=rows[0];pairs=[encode_pair(tok,*p) for p in row['request_pairs']]
  native=scorer.score_uncached(pairs);model.early_exit_step=3
  reference=HFLM(pretrained=model,tokenizer=tok,backend='causal',batch_size=1,device='cuda',max_length=32768,add_bos_token=False,dtype='bfloat16',trust_remote_code=True)
  instances=[Instance(request_type='loglikelihood',doc=row['source_doc'],arguments=tuple(pair),idx=i,metadata=('arc_challenge',row['source_index'],1)) for i,pair in enumerate(row['request_pairs'])]
  expected=reference.loglikelihood(instances,disable_tqdm=True)
  actual=[r['sum_loglikelihood'][4] for r in native]
  errors=[abs(e[0]-x)/len(t) for e,x,(_,t) in zip(expected,actual,pairs)]
  assert max(errors)<=.05
  expected_pred=int(np.argmax([e[0]/max(n,1) for e,n in zip(expected,row['choice_char_lengths'])]));actual_pred=outcome(actual,row['choice_char_lengths'])['prediction'];assert expected_pred==actual_pred
  save(str(parity_path.relative_to(ROOT)),dict(status='passed',cases=cases,harness_k4_max_error_per_token=max(errors),harness_prediction_matches=True))
 grid=[0,.125,.25,.5,1.]
 def readout(state,target):
  z=model.lm_head(state)[0].float();t=torch.tensor(target,device=z.device)
  return float(z.log_softmax(-1).gather(-1,t[:,None]).sum())
 for row in rows:
  if row['id'] in completed:continue
  check_budget(90);pairs=[encode_pair(tok,*p) for p in row['request_pairs']]
  assert all(p[:len(shared_ids)]==shared_ids for p,t in pairs)
  torch.cuda.synchronize();beg=time.time()
  if a.uncached:scored=scorer.score_uncached(pairs,capture_raw=True,return_states=True)
  else:scored=scorer.score_candidates(pairs,shared=shared,capture_raw=True,return_states=True)
  torch.cuda.synchronize();native_seconds=time.time()-beg
  lengths=row['choice_char_lengths'];gold=row['gold']
  methods={f'native_k{k}':outcome([s['sum_loglikelihood'][k] for s in scored],lengths,gold) for k in range(1,7)}
  for name in ['fixed','adaptive']:methods['loopcd_k4_'+name]=outcome([s['loopcd'][name] for s in scored],lengths,gold)
  paths={f'{kind}_{lam:g}':[] for kind in ['normalized_chord','raw_chord'] for lam in grid}
  geometry=[];payload=[]
  for j,(s,(_,targets)) in enumerate(zip(scored,pairs)):
   states=s['normalized_states'];raw=s['raw_states'];assert len(states)==len(raw)==6
   payload.append(dict(candidate=j,target_ids=targets,normalized=states,raw=raw))
   h4,h5=[states[k].to('cuda') for k in [3,4]];r4,r5=[raw[k].to('cuda') for k in [3,4]]
   delta=h5.float()-h4.float()
   geometry.append(dict(candidate=j,token_count=len(targets),h4_to_h5_l2=float(delta.norm()),
      relative_l2=float(delta.norm()/h4.float().norm().clamp_min(1e-12)),
      cosine=float(torch.nn.functional.cosine_similarity(h4.float(),h5.float(),dim=-1).mean())))
   for lam in grid:
    if lam in [0,1]:value=s['sum_loglikelihood'][4 if lam==0 else 5]
    else:value=readout((h4.float()+lam*delta).to(torch.bfloat16),targets)
    paths[f'normalized_chord_{lam:g}'].append(value)
    point=(r4.float()+lam*(r5.float()-r4.float())).to(torch.bfloat16)
    paths[f'raw_chord_{lam:g}'].append(readout(model.model.norm(point),targets))
  for name,values in paths.items():methods[name]=outcome(values,lengths,gold)
  stem=row['id'].split('/')[-1];torch.save(dict(id=row['id'],candidates=payload),out/(stem+'_states.pt'))
  result=dict(id=row['id'],source_index=row['source_index'],gold=gold,scope='official_train_discovery_only',
    methods=methods,geometry=geometry,choice_char_lengths=lengths,choice_token_counts=[len(t) for p,t in pairs],
    prompt_sha256=row['prompt_sha256'],fewshot=25,demonstrations='fixed_global_25',cached=not a.uncached,
    implementation_sha256=codehash,native_scoring_seconds=native_seconds,total_case_seconds=time.time()-beg)
  with path.open('a',encoding='utf8') as f:f.write(json.dumps(result)+'\n')
  completed[row['id']]=result
  cohort=[completed[x['id']] for x in rows if x['id'] in completed]
  progress=dict(completed=len(cohort),target=a.n,elapsed_seconds=time.time()-start,
    accuracy={name:sum(x['methods'][name]['correct'] for x in cohort)/len(cohort) for name in ['native_k3','native_k4','native_k5','native_k6']})
  save(str(out.relative_to(ROOT)/'PROGRESS.json'),progress)
  if len(cohort)%4==0:print(json.dumps(progress),flush=True)
 cohort=[completed[x['id']] for x in rows]
 seconds=[x['native_scoring_seconds'] for x in cohort]
 forecast=dict(n=a.n,native_mean_seconds=float(np.mean(seconds)),native_max_seconds=max(seconds),prefill_seconds=prefill_seconds,
    conservative_two_complete_test_trajectories_seconds=2*1172*float(np.mean(seconds))*1.5+2*prefill_seconds,
    note='Loaded-worker budget also includes parity/analysis; adjust diagnosis breadth before final frozen evaluation, never shorten1172 test')
 save(str(out.relative_to(ROOT)/'BUDGET_FORECAST.json'),forecast)
 save(str(out.relative_to(ROOT)/f'COMPLETE_{a.n}.json'),dict(status='complete',n=a.n,elapsed_seconds=time.time()-start,implementation_sha256=codehash))
 print(json.dumps(forecast),flush=True)

if __name__=='__main__':main()
