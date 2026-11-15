"""Build the Sol-verified C1 packet for scientific direction; not a completion claim."""
import argparse,collections,gzip,hashlib,json,math
from pathlib import Path
import numpy as np
import torch
from C1_AUDIT_TENSOR_IO import load_checked

ROOT=Path('/root/autodl-tmp/ouro-k1-k2-mechanism')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def rows(p):
 with gzip.open(p,'rt') as f:return [json.loads(line) for line in f]
def gap(scores,gold):return scores[gold]-sum(v for i,v in enumerate(scores) if i!=gold)/(len(scores)-1)
def sign(v):return int(v>0)-int(v<0)
def main():
 p=argparse.ArgumentParser();p.add_argument('--numeric-root',required=True);args=p.parse_args()
 numeric_root=ROOT/args.numeric_root
 assert numeric_root.resolve().is_relative_to(ROOT.resolve())
 a=ROOT/'analysis';out=ROOT/'evidence_packet';assert not out.exists()
 assert read(ROOT/'C1_QUEUE_STATE.json')['status']=='COMPLETE'
 assert read(numeric_root/'COMPLETE.json')['n']==8
 train=rows(a/'TRAIN_QUESTION_CAUSAL_EVIDENCE.jsonl.gz');val=rows(a/'VALIDATION_QUESTION_CAUSAL_EVIDENCE.jsonl.gz')
 assert len(train)==1094 and len(val)==299
 allq={q['id']:q for q in train+val};assert len(allq)==1393
 ids64=set(read(ROOT/'inputs/BATCH_IDS.json')['train64'])
 cohorts={'train64':[q for q in train if q['id'] in ids64],
          'train1030':[q for q in train if q['id'] not in ids64],
          'train1094':train,'validation299':val}
 for split,n in [('TRAIN',1094),('VALIDATION',299)]:
  for suffix in ['COMPLETE.json','ALL_ENDPOINT_CPU_AUDIT.json','ALL_CONTROL_TENSOR_AUDIT.json']:
   assert read(a/(split+'_'+suffix))['status']=='PASS'
 summary={}
 for label,qs in cohorts.items():
  summary[label]={'n':len(qs),'natural_groups':dict(collections.Counter(q['natural_group'] for q in qs)),
    'candidate_counts':dict(collections.Counter(len(q['endpoints']['K1']['candidate_logits']) for q in qs)),
    'correct':{name:sum(q['endpoints'][name]['correct'] for q in qs) for name in ['K1','N','P','P_plus_L','U','UX','UA','UXA']},
    'native_tie_count':{name:sum(len(q['endpoints'][name]['top_ties'])>1 for q in qs) for name in ['K1','N','U','UX','UA','UXA']}}
 tables=read(a/'TRAIN_CONTRAST_TABLES.json')+read(a/'VALIDATION_CONTRAST_TABLES.json')
 primary=[r for r in tables if r['group']=='rescue' and r['sensitivity']=='all' and r['metric']=='meanwrong']
 numeric=[];max_error=0.
 subset=read(ROOT/'C1_FP32_NUMERIC_SELECTION.json')['selected_ids']
 for ident in subset:
  p=numeric_root/(hashlib.sha256(ident.encode()).hexdigest()+'.json');m=read(p)
  raw,storage=load_checked(p.with_suffix('.safetensors'),m);q=allq[ident];gold=q['gold_offline_only']
  endpoints={};floatgaps={};bf16gaps={}
  for name in ['K1','N','P','U','UX','UA','UXA']:
   tensor=raw[name+'_logits'];assert tensor.dtype==torch.float32 and tensor.numel()==49152
   digest=hashlib.sha256(tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
   c=m['conditions'][name];assert digest==c['Rlogits_sha256']
   v=tensor.reshape(-1).numpy().astype(np.float64);target=m['targets'];candidates=v[target].tolist()
   assert candidates==c['candidate_logits']
   mx=float(v.max());lse=mx+math.log(float(np.exp(v-mx).sum(dtype=np.float64)))
   deviation=float(np.max(np.abs(np.asarray(candidates)-lse-np.asarray(c['scores']))))
   envelope=8*np.finfo(np.float32).eps*(1+float(np.abs(v).max())+abs(lse));assert deviation<=envelope
   max_error=max(max_error,deviation)
   tie=[i for i,x in enumerate(candidates) if x==max(candidates)]
   assert tie==c['top_ties'] and tie[0]==c['prediction']
   b=q['endpoints'][name];floatgaps[name]=gap(c['scores'],gold);bf16gaps[name]=b['meanwrong']
   endpoints[name]={'BF16_prediction':b['prediction'],'FP32_prediction':c['prediction'],
       'BF16_top_ties':b['top_ties'],'FP32_top_ties':tie,'prediction_changed':b['prediction']!=c['prediction'],
       'BF16_correct':b['correct'],'FP32_correct':c['prediction']==gold,
       'BF16_gap':bf16gaps[name],'FP32_gap':floatgaps[name],
       'independent_FP64_normalizer_error':deviation,'a_priori_FP32_envelope':envelope}
  def core(v):
   return {'total_upstream_loss':v['N']-v['U'],'X_restore':v['UX']-v['U'],
      'X_loss':v['N']-v['UA'],'A_restore':v['UA']-v['U'],'A_loss':v['N']-v['UX'],
      'interaction':v['N']-v['UX']-v['UA']+v['U']}
  bg=core(bf16gaps);fg=core(floatgaps)
  for name,v in bg.items():assert abs(v-q['contrasts'][name]['meanwrong'])<1e-12
  numeric.append({'id':ident,'native_BF16_group':q['natural_group'],'metadata_sha256':sha(p),'tensor_storage':storage,
    'endpoints':endpoints,'contrasts':{name:{'BF16':v,'FP32':fg[name],'sign_changed':sign(v)!=sign(fg[name])} for name,v in bg.items()},
    'actual_SDPA_dispatch_events':m['actual_SDPA_profiler_events']})
 counter={}
 for label,qs in cohorts.items():
  categories={
   'natural_rescue_blocked_not_recovered_by_either_single_branch':lambda q:q['natural_group']=='rescue' and not q['endpoints']['U']['correct'] and not q['endpoints']['UX']['correct'] and not q['endpoints']['UA']['correct'],
   'natural_rescue_not_blocked_by_upstream_intervention':lambda q:q['natural_group']=='rescue' and q['endpoints']['U']['correct'],
   'native_correct_damaged_by_X_restoration':lambda q:q['endpoints']['N']['correct'] and not q['endpoints']['UX']['correct'],
   'native_correct_damaged_by_A_restoration':lambda q:q['endpoints']['N']['correct'] and not q['endpoints']['UA']['correct'],
   'X_recovery_fails_strict_controls_or_ties':lambda q:q['branch_decisions']['X']['recovered_from_U'] and not q['branch_decisions']['X']['strict_controlled_recovery'],
   'A_recovery_fails_strict_controls_or_ties':lambda q:q['branch_decisions']['A']['recovered_from_U'] and not q['branch_decisions']['A']['strict_controlled_recovery'],
   'strict_X_bidirectional':lambda q:q['branch_decisions']['X']['strict_controlled_bidirectional'],
   'strict_A_bidirectional':lambda q:q['branch_decisions']['A']['strict_controlled_bidirectional']}
  counter[label]={key:sorted(q['id'] for q in qs if condition(q)) for key,condition in categories.items()}
 out.mkdir();write(out/'COHORT_NATIVE_AND_CORNER_COUNTS.json',summary)
 write(out/'PRIMARY_RESCUE_CONTRASTS.json',primary);write(out/'NUMERICAL8_INDEPENDENT_COMPARISON.json',
   {'status':'PASS','max_FP32_score_error_vs_FP64':max_error,'scope':'Combined numerical/backend sensitivity on fixed8, with independent full-vocabulary normalizer/tie audit; not an isolated dtype effect, model-seed replication or a retuned subset.','cases':numeric})
 write(out/'ALL_COUNTEREXAMPLE_AND_STRICT_CASE_IDS.json',counter)
 gate=read(a/'C1_ROUTE_GATE_VERDICT.json');write(out/'C1_DIRECTOR_INPUT.json',
   {'scope':'C1 local upstream-to-L18 branch evidence. Overall K1-to-K2 goal remains open; boundary-to-Q9 generation is not resolved by this packet.',
    'cohorts':summary,'fixed_route_gate':gate,'primary_tables':'PRIMARY_RESCUE_CONTRASTS.json',
    'all_question_evidence':['../analysis/TRAIN_QUESTION_CAUSAL_EVIDENCE.jsonl.gz','../analysis/VALIDATION_QUESTION_CAUSAL_EVIDENCE.jsonl.gz'],
    'numeric8':'NUMERICAL8_INDEPENDENT_COMPARISON.json','numeric_execution_root':args.numeric_root,
    'numerical_history':'Original FP32 efficient-attention execution failed its strict same-input donor equality before any P intervention. It remains preserved in numeric_fp32/. Separate deterministic+math sensitivity uses identical preselected8 with fresh computations; changes combine backend and determinism configuration, not isolated dtype effects. See FP32_NATIVE_REPEAT_DIAGNOSIS.md.',
    'counterexamples':'ALL_COUNTEREXAMPLE_AND_STRICT_CASE_IDS.json',
    'prior_E3_control_scope':'New C1 sorted fullsplit cyclic upstream unrelated check; not exact old4090 pool/mapping replication.',
    'failure_history':'Original scalar serialization failure and chronology correction retained. Scientific conditions/cohorts never changed from outcomes.'})
 text='# C1 natural correction mediation: interim Sol evidence\n\nThis is the completed fixed C1 experiment, not completion of the full K1→K2 investigation. All four natural groups and frozen controls are retained. Train64 is the old engineering/discovery cohort; it partitions fullTrain1094 with remaining1030 and is not a population estimate or an independent replication. Validation299 was historically exposed and is reported separately.\n\n'
 text+='| Cohort | N | K1 correct | NativeK2 | U | RestoreX | RestoreA | Joint structural anchor |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
 for label,s in summary.items():
  c=s['correct'];text+=f"| {label} | {s['n']} | {c['K1']} | {c['N']} | {c['U']} | {c['UX']} | {c['UA']} | {c['UXA']} |\n"
 text+='\nPrimary estimates use production FP32 log-probability gold-minus-mean-wrong gaps on fresh natural rescues. The machine-readable tables contain all raw/control-adjusted contrasts, paired intervals, simultaneous one-sided12-test bounds, defined coverage and exact BF16 Fraction signs. No contrast was selected after observing results.\n\n'
 text+='| Cohort | Contrast | N defined/population | Mean | Paired95% | Family12 one-sided lower |\n|---|---|---:|---:|---|---:|\n'
 for row in primary:
  if row['cohort'] not in ['train1030','validation299']:continue
  text+=f"| {row['cohort']} | {row['contrast']} | {row['n_defined']}/{row['n_population']} | {row['mean']} | {row['paired95']} | {row['one_sided95_lower_Bonferroni12']} |\n"
 text+='\nFrozen strong-route decisions:\n\n'+json.dumps(gate,indent=2)+'\n\nJoint restoration is an expected structural endpoint identity in matched N/U non-R contexts, not a discovered mediated percentage. The upstream operator is Q9-conditioned N−P reference subtraction, not natural K2−K1. The full recurrent-boundary generation edge remains unresolved.\n'
 (ROOT/'NATURAL_MEDIATION_RESULTS.md').write_text(text)
 text='# C1 counterexamples and strict cases\n\nAll IDs are retained in evidence_packet/ALL_COUNTEREXAMPLE_AND_STRICT_CASE_IDS.json and the per-question raw evidence. The examples below are the first10 lexicographic IDs for each predeclared symmetric category, never selected for a favorable effect.\n\n'
 for label,categories in counter.items():
  text+='## '+label+'\n\n'
  for category,ids in categories.items():text+=f"- {category}: N={len(ids)}; "+', '.join(ids[:10])+'\n'
  text+='\n'
 (ROOT/'COUNTEREXAMPLES.md').write_text(text)
 files=[{'file':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in sorted(out.rglob('*')) if p.is_file()]
 write(out/'PACKET_MANIFEST.json',{'source_sha256':sha(__file__),'files':files,
       'BF16_queue_state_sha256':sha(ROOT/'C1_QUEUE_STATE.json'),'numeric_complete_sha256':sha(numeric_root/'COMPLETE.json')})
 print('C1_EVIDENCE_PACKET_COMPLETE',len(train),len(val),len(numeric),flush=True)
if __name__=='__main__':main()
