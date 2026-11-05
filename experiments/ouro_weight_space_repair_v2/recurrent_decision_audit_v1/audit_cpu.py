"""Historical descriptive R11 audit; read-only parent artifacts, no model import."""
from pathlib import Path
import sys,json,hashlib,csv,time,collections,platform
import numpy as np
from scipy.stats import binomtest
ROOT=Path(__file__).resolve().parent.parent
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
from r7_core import digest
from r10_core import joint_prefix,joint_requests
from r11_evaluate import records
import r7_core
class ReadOnlyOverlay:
 def __truediv__(self,p):
  original=ROOT/p
  return original if original.exists() else OUT/'recovered'/p
r7_core.ROOT=ReadOnlyOverlay()

def read(p):return json.loads((ROOT/p).read_text(encoding='utf8'))
def lines(p):return [json.loads(s) for s in (ROOT/p if (ROOT/p).exists() else OUT/'recovered'/p).read_text(encoding='utf8').splitlines()]
def sha(p):return hashlib.sha256((ROOT/p if (ROOT/p).exists() else OUT/'recovered'/p).read_bytes()).hexdigest()
def save(p,x):(OUT/p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
def csvout(p,rows):
 with (OUT/p).open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def pair(tok,c,a):
 n=len(c)-len(c.rstrip())
 if n:a=c[-n:]+a;c=c[:-n]
 p=tok.encode(c,add_special_tokens=False).ids
 return p,tok.encode(c+a,add_special_tokens=False).ids[len(p):]
def bootstrap(x,seed=2021):
 x=np.asarray(x,float);rng=np.random.default_rng(seed);v=[]
 for _ in range(200):v.extend(x[rng.integers(0,len(x),(100,len(x)))].mean(1).tolist())
 return np.quantile(v,[.025,.975]).tolist()
def cond_boot(stable,correct):
 stable=np.asarray(stable,bool);correct=np.asarray(correct,bool);rng=np.random.default_rng(2021);v=[]
 for _ in range(200):
  ix=rng.integers(0,len(stable),(100,len(stable)));s=stable[ix];c=correct[ix];ns=s.sum(1);nu=(~s).sum(1);ok=(ns>0)&(nu>0)
  v.extend(((c&s).sum(1)[ok]/ns[ok]-(c&~s).sum(1)[ok]/nu[ok]).tolist())
 return np.quantile(v,[.025,.975]).tolist() if v else None

def main():
 started=time.monotonic();from r11_core import gate;gate('test');assert sha('results/r11_test_selected.jsonl')==read('results/r11_test_selected_COMPLETE.json')['score_file_sha256'];data=lines('results/r11_test_selected.jsonl') # source-bound raw receipt; original-environment official replay is separate
 assert len(data)==1172
 items={r['id']:r for r in lines('data/slcd_test.jsonl')};prefix=joint_prefix(read('data/slcd_fixed_demonstrations.json'))
 assert hashlib.sha256(prefix.encode()).hexdigest()==read('configs/r10_PROTOCOL.json')['joint_demonstrations_sha256']
 from tokenizers import Tokenizer
 import argparse;ap=argparse.ArgumentParser();ap.add_argument('--tokenizer-json');args=ap.parse_args();tp=args.tokenizer_json or next(str(x) for x in [ROOT/'models/Ouro-1.4B/tokenizer.json',ROOT/'artifacts/r24_TOKENIZER_AUDIT_PRIVATE.json'] if x.exists());assert sha(tp)=='fcb808fe5e7642f5299be28aea07fc7f6d4f4364c3ac5e408e15a772cbc8fa8d'
 tok=Tokenizer.from_file(str(ROOT/tp));tokenchecks=0;requestschecked=0;tie=collections.Counter();predrows=[];rotationrows=[];stabilityrows=[];transitionrows=[];pairsummary=[]
 for r in data:
  item=items[r['id']];assert item['gold']==r['gold'];n=len(item['choices']);
  for key,vec in {**r['baseline_scores'],**r['method_scores'],**r['joint_baseline_scores']}.items():assert int(np.asarray(vec).argmax())==r['predictions'][key] and r['correct'][key]==(r['predictions'][key]==r['gold'])
  for k in [1,4]:
   matrix=np.asarray([x['sum_scores'][f'native_k{k}'] for x in r['cyclic_rotations']]);assert np.array_equal(matrix[0],r['joint_baseline_scores'][f'joint{k}_only']) and np.array_equal(matrix.mean(0),r['joint_baseline_scores'][f'cyclic_joint{k}_only'])
  assert item['choice_char_lengths']==[len(t) for t in item['choices']]
  assert digest(item['prompt'])==r['prompt_sha256'] or hashlib.sha256(item['prompt'].encode()).hexdigest()==r['prompt_sha256']
  counts=[];lengths=[];hashes=[];mappedpred={1:[],4:[]}
  for shift,rot in enumerate(r['cyclic_rotations']):
   order=rot['order'];req=joint_requests(dict(question=item['question'],choices=[item['choices'][i] for i in order]),prefix)
   assert [digest(x) for x in req]==rot['request_sha256'];requestschecked+=len(req)
   for c,a in req:
    p,t=pair(tok,c,a);assert p and t;counts.append(len(t));lengths.append(len(p)+len(t)-1);hashes.append(digest(t));tokenchecks+=1
   for k in [1,4]:
    mapped=np.asarray(rot['sum_scores'][f'native_k{k}'],float);assert mapped.shape==(n,) and np.isfinite(mapped).all()
    displayed=mapped[order];j=int(displayed.argmax());pred=order[j];margmax=int(mapped.argmax());ntie=int((displayed==displayed.max()).sum());tie[f'K{k}_rotation_ties']+=ntie>1;tie[f'K{k}_mapped_argmax_diff']+=pred!=margmax
    mappedpred[k].append(pred)
    rotationrows.append(dict(id=r['id'],K=k,shift=shift,order=json.dumps(order),displayed_prediction=j,original_content_prediction=pred,mapped_argmax_prediction=margmax,tied_max_count=ntie,mapped_scores=json.dumps(mapped.tolist())))
  assert counts==r['target_token_counts'] and lengths==r['model_input_token_counts'] and hashes==r['target_ids_sha256']
  for k in [1,4]:
   matrix=np.asarray([x['sum_scores'][f'native_k{k}'] for x in r['cyclic_rotations']],float);ps=mappedpred[k];unanimous=len(set(ps))==1
   agree=sum(ps[i]==ps[j] for i in range(n) for j in range(i+1,n))/(n*(n-1)/2)
   stabilityrows.append(dict(id=r['id'],K=k,nchoices=n,unanimous=unanimous,changed_across_rotations=not unanimous,pairwise_content_agreement=agree,unique_content_predictions=len(set(ps)),mapped_candidate_variances=json.dumps(matrix.var(axis=0,ddof=0).tolist()),mean_mapped_score_variance=float(matrix.var(axis=0,ddof=0).mean()),mean_centered_score_variance=float((matrix-matrix.mean(axis=1,keepdims=True)).var(axis=0,ddof=0).mean()),original_correct=r['correct'][f'joint{k}_only'],cyclic_mean_correct=r['correct'][f'cyclic_joint{k}_only']))
  for family,names in [('joint',('joint1_only','joint4_only')),('cyclic_mean',('cyclic_joint1_only','cyclic_joint4_only'))]:
   p1,p4=[r['predictions'][x] for x in names];c1,c4=[r['correct'][x] for x in names]
   transition='correct_correct' if c1 and c4 else 'rescue' if c4 else 'damage' if c1 else 'wrong_wrong'
   transitionrows.append(dict(id=r['id'],family=family,gold=r['gold'],K1_prediction=p1,K4_prediction=p4,K1_correct=c1,K4_correct=c4,transition=transition,wrong_wrong_changed=not c1 and not c4 and p1!=p4,prediction_unchanged=p1==p4))
   for k,pred,c in [(1,p1,c1),(4,p4,c4)]:
    vec=np.asarray(r['joint_baseline_scores'][names[0 if k==1 else 1]],float);margin=float(vec[r['gold']]-np.delete(vec,r['gold']).max())
    predrows.append(dict(id=r['id'],family=family,K=k,prediction=pred,gold=r['gold'],correct=c,gold_margin=margin,first_observed_correct_K=1 if c1 else 4 if c4 else '',first_observed_change_K=4 if p1!=p4 else '',endpoint_stability_confirmed_at_K=4 if p1==p4 else '',endpoint_gap=3,observed_endpoint_reversals='',unobserved_intermediate_reversals='unknown',true_first_correction_interval='K2-K4' if not c1 and c4 else '',scores=json.dumps(vec.tolist())))
 for family in ['joint','cyclic_mean']:
  rr=[r for r in transitionrows if r['family']==family];res=sum(r['transition']=='rescue' for r in rr);dam=sum(r['transition']=='damage' for r in rr);v=[int(r['K4_correct'])-int(r['K1_correct']) for r in rr]
  pairsummary.append(dict(family=family,n=len(rr),K1_correct=sum(r['K1_correct'] for r in rr),K4_correct=sum(r['K4_correct'] for r in rr),rescue=res,damage=dam,correct_correct=sum(r['transition']=='correct_correct' for r in rr),wrong_wrong=sum(r['transition']=='wrong_wrong' for r in rr),wrong_wrong_changed=sum(r['wrong_wrong_changed'] for r in rr),prediction_unchanged=sum(r['prediction_unchanged'] for r in rr),net=res-dam,exact_mcnemar_p=float(binomtest(min(res,dam),res+dam,.5).pvalue),bootstrap_95_delta_accuracy=bootstrap(v),bootstrap_seed=2021,bootstrap_samples=20000))
 assert [x['K1_correct'] for x in pairsummary]==[634,657] and [x['K4_correct'] for x in pairsummary]==[1021,1036]
 assert [x['net'] for x in pairsummary]==[387,379]
 stability=[]
 for k in [1,4]:
  ss=[x for x in stabilityrows if x['K']==k];s=np.asarray([x['unanimous'] for x in ss]);c=np.asarray([x['original_correct'] for x in ss]);stablecorrect=int((s&c).sum());unstablecorrect=int((~s&c).sum())
  stability.append(dict(K=k,questions=len(ss),unanimous=int(s.sum()),changed=int((~s).sum()),mean_pairwise_agreement=float(np.mean([x['pairwise_content_agreement'] for x in ss])),mean_score_variance=float(np.mean([x['mean_mapped_score_variance'] for x in ss])),stable_correct=stablecorrect,stable_n=int(s.sum()),unstable_correct=unstablecorrect,unstable_n=int((~s).sum()),stable_accuracy=stablecorrect/int(s.sum()) if s.any() else None,unstable_accuracy=unstablecorrect/int((~s).sum()) if (~s).any() else None,stable_minus_unstable_bootstrap95=cond_boot(s,c)))
 s1=[x for x in stabilityrows if x['K']==1];s4=[x for x in stabilityrows if x['K']==4];assert [x['id'] for x in s1]==[x['id'] for x in s4]
 diff={key:dict(mean=float(np.mean([b[key]-a[key] for a,b in zip(s1,s4)])),paired_bootstrap95=bootstrap([b[key]-a[key] for a,b in zip(s1,s4)])) for key in ['pairwise_content_agreement','mean_mapped_score_variance','mean_centered_score_variance']}
 save('existing_score_parity.json',dict(status='passed_hash_and_independent_joint_request_token_replay',n=1172,official_restore_log='official_restore_verify.log',raw_score_sha256=sha('results/r11_test_selected.jsonl'),request_hashes_checked=requestschecked,target_token_hashes_counts_checked=tokenchecks,original_character_denominators_checked=True,joint_letter_scores_are_unnormalized_target_sums=True,cyclic_mean_is_mean_of_mapped_sum_scores=True,counts={k:sum(r['correct'][k] for r in data) for k in data[0]['correct']},historical_model_identity=read('configs/r11_PROTOCOL.json')['model_identity'],live_model_loaded=False))
 csvout('k1_k4_paired_transitions.csv',transitionrows);csvout('loopwise_predictions.csv',predrows);csvout('loopwise_rescue_damage.csv',[dict(x,bootstrap_95_delta_accuracy=json.dumps(x['bootstrap_95_delta_accuracy'])) for x in pairsummary]);csvout('permutation_stability.csv',stabilityrows);csvout('cyclic_rotation_predictions.csv',rotationrows)
 save('paired_statistics.json',pairsummary);save('stability_summary.json',dict(per_depth=stability,paired_K4_minus_K1=diff,ties=dict(tie),nchoices_distribution=dict(collections.Counter(len(items[r['id']]['choices']) for r in data)),scope='Question-level 20k bootstrap. Rotation scores already mapped to original contents; displayed argmax then mapped via order. Population variance ddof0 of target LP sum, not probability.'))
 coverage=dict(joint_test={'questions':1172,'available_native_K':[1,4],'unavailable_K':[2,3,5,6],'source_reason':'r7_inference.head_logits selects (1,4,5), infer_auxiliary depth4; no joint states persisted in record archive.'},cyclic_rotations='all n cyclic rotations, not full permutation group',ordinary_nonjoint_cache_not_compatible=True,trajectory_scope='Only observed endpoints. Equal endpoints cannot exclude intermediate reversals. First actual correction among K2..K4 is unresolved.',stable_criterion='equal predictions at adjacent available endpoints K1/K4, gap3; not consecutive actual depths',intermediate_imputation=False)
 save('depth_readout_coverage.json',coverage)
 ledgernames=['r7','r9','r10','r11','r15','r16','r17','r18','r19','r20','r20_e1','r20_e2','r23','r25','r26','r27_EXEC','r28','r30','r33'];ledgers=[]
 for n in ledgernames:
  path='logs/'+n+'_gpu_budget.json';v=read(path);ledgers.append(dict(path=path,sha256=sha(path),used_seconds=v['used_seconds']))
 total=sum(x['used_seconds'] for x in ledgers);assert total==20834.389865819365
 save('compute_ledger.json',dict(inherited=ledgers,inherited_used_seconds=total,cap_seconds=21600,remaining_seconds=21600-total,new_GPU_worker_seconds=0,CPU_audit_seconds=time.monotonic()-started,model_loaded=False,python=sys.version,platform=platform.platform()))
 save('input_provenance.json',dict(files={p:sha(p) for p in ['scripts/r10_core.py','scripts/r11_inference.py','scripts/r11_evaluate.py','scripts/r7_inference.py','scripts/ouro_cached_scoring.py','scripts/runtime.py','artifacts/r11_RECORDS_INDEX.json','artifacts/r11_RECORDS_CONTENTS.json','configs/r11_PROTOCOL.json','configs/r11_FINAL_METHOD_LOCK.json','data/slcd_test.jsonl','data/slcd_fixed_demonstrations.json','results/r11_test_selected.jsonl','results/r11_test_selected_COMPLETE.json']},audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
 print(json.dumps(dict(paired=pairsummary,stability=stability,differences=diff,ties=dict(tie)),indent=2))
if __name__=='__main__':main()
