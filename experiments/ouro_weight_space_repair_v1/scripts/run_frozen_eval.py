"""One frozen full-sequence trajectory per arm. No hypothesis generation."""
import argparse,hashlib,json,time
from runtime import ROOT,load_model,encode_pair,save,check_budget,outcome
from ouro_cached_scoring import OuroCachedScorer

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--split',required=True,choices=['selection','validation','test']);ap.add_argument('--recipe',required=True);a=ap.parse_args()
 recipe_path=ROOT/'configs'/a.recipe;recipe=json.loads(recipe_path.read_text())
 assert recipe['candidate_definitions_frozen'] is True and recipe['source_of_directions']!='test'
 if a.split=='test':
  lock=json.loads((ROOT/'configs/FINAL_LOCK.json').read_text());assert lock['test_evaluation_authorized'] is True and lock['recipe_sha256']==digest(recipe_path)
 if a.split=='validation':
  lock=json.loads((ROOT/'configs/CANDIDATE_LOCK.json').read_text());assert lock['recipe_sha256']==digest(recipe_path)
 manifest_path=ROOT/'data'/f'{a.split}.jsonl';manifest_sha=digest(manifest_path)
 gate=json.loads((ROOT/'results/READINESS.json').read_text());assert gate.get('strict_zero_error_original_parity') is True
 rows=[json.loads(l) for l in manifest_path.read_text().splitlines()]
 expected={'selection':192,'validation':299,'test':1172}[a.split];assert len(rows)==expected
 out=ROOT/'results'/('frozen_'+a.split);out.mkdir(exist_ok=True)
 codehash=hashlib.sha256(''.join(digest(ROOT/'scripts'/n) for n in ['runtime.py','ouro_cached_scoring.py','run_frozen_eval.py']).encode()).hexdigest()
 completed={};path=out/'scores.jsonl'
 if path.exists():
  for line in path.read_text().splitlines():
   r=json.loads(line);assert r['implementation_sha256']==codehash and r['recipe_sha256']==digest(recipe_path) and r['data_sha256']==manifest_sha;assert r['id'] not in completed;completed[r['id']]=r
 start=time.time();tok,model=load_model();scorer=OuroCachedScorer(model,6)
 import torch
 def readout(state,target):
  z=model.lm_head(state)[0].float();t=torch.tensor(target,device=z.device)
  return float(z.log_softmax(-1).gather(-1,t[:,None]).sum())
 assert recipe['weight_edits']==[], 'Only explicit no-edit path implemented; add exact patch runner before freezing a real edit'
 for row in rows:
  if row['id'] in completed:continue
  check_budget(60);pairs=[encode_pair(tok,*p) for p in row['request_pairs']]
  began=time.time();scored=scorer.score_uncached(pairs,capture_raw=False,return_states=True)
  m={f'native_k{k}':outcome([s['sum_loglikelihood'][k] for s in scored],row['choice_char_lengths'],row['gold']) for k in range(1,7)}
  for name in ['fixed','adaptive']:m['loopcd_k4_'+name]=outcome([s['loopcd'][name] for s in scored],row['choice_char_lengths'],row['gold'])
  for lam in recipe['readout_chord_coefficients']:
   assert 0<float(lam)<1
   sums=[]
   for s,(_,targets) in zip(scored,pairs):
    h4,h5=[s['normalized_states'][k].to('cuda') for k in [3,4]]
    point=(h4.float()+float(lam)*(h5.float()-h4.float())).to(torch.bfloat16)
    sums.append(readout(point,targets))
   m[f'normalized_chord_{lam:g}']=outcome(sums,row['choice_char_lengths'],row['gold'])
  r=dict(id=row['id'],source_index=row['source_index'],split=a.split,gold=row['gold'],methods=m,prompt_sha256=row['prompt_sha256'],choice_char_lengths=row['choice_char_lengths'],choice_token_counts=[len(t) for _,t in pairs],fewshot=25,demonstrations='fixed_global_25',cached=False,implementation_sha256=codehash,recipe_sha256=digest(recipe_path),data_sha256=manifest_sha,elapsed_seconds=time.time()-began)
  with path.open('a',encoding='utf8') as f:f.write(json.dumps(r)+'\n')
  completed[row['id']]=r
  if len(completed)%8==0 or len(completed)==len(rows):
   progress=dict(completed=len(completed),target=len(rows),elapsed_seconds=time.time()-start,split=a.split)
   save(str(out.relative_to(ROOT)/'PROGRESS.json'),progress);print(json.dumps(progress),flush=True)
 save(str(out.relative_to(ROOT)/'COMPLETE.json'),dict(status='complete',n=len(completed),expected_n=expected,recipe_sha256=digest(recipe_path),data_sha256=manifest_sha,implementation_sha256=codehash,elapsed_seconds=time.time()-start))
if __name__=='__main__':main()
