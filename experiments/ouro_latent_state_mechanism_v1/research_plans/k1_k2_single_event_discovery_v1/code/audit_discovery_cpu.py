"""All saved head endpoints, scalar contracts and independently recomputed selection. Sol self-audit."""
import pathlib,json,hashlib,datetime,torch,numpy as np
from fractions import Fraction as F
from guard import verify,sha,exact_ids
R=pathlib.Path('/root/ouro-binding-interface-sol-20261117');RAW=pathlib.Path('/root/autodl-tmp/ouro-binding-interface-sol-20261117/raw');P=pathlib.Path('/root/ouro-semantic-read-sol-20261117');torch.set_num_threads(4);fs=verify();qs={q['id']:q for q in map(json.loads,(R/'DEV_INPUTS.jsonl').read_text().splitlines())};count=0;maxlp=0.;rows=[];events=0;maxrecon=0.
def endpoint(x,s,q):
 global count,maxlp
 assert x.numel()==49152 and x.dtype==torch.bfloat16
 assert hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()==s['full49152_sha256']
 v=x.flatten().double().numpy();c=v[q['targets']];assert c.tolist()==s['candidate_logits'];ties=np.flatnonzero(c==c.max()).tolist();assert ties==s['ties'] and ties[0]==s['prediction'];m=float(v.max());lp=c-m-np.log(np.exp(v-m).sum());err=float(abs(lp-np.array(s['candidate_logprobs'])).max());assert err<2e-5;maxlp=max(maxlp,err);count+=1
for p in sorted((R/'results').glob('SR*_L*.json')):
 z=json.loads(p.read_text());assert z['freeze_sha256']==fs;layer=z['layer'];exact_ids(z['rows'],[q['id'] for q in qs.values() if q['pair_id']==z['pair_id']]);rp=RAW/(p.stem+'.pt');assert sha(rp)==z['raw_sha256'];raw=torch.load(rp,map_location='cpu',weights_only=False);assert set(raw)=={u['id'] for u in z['rows']};nd={u['id']:u for u in json.loads((P/'native_results'/(z['pair_id']+'.json')).read_text())['rows']}
 for u in z['rows']:
  q=qs[u['id']];assert u['native']==nd[u['id']]['native'];assert all(u[k]==q[k] for k in ['pair_id','template','firsthop_factor','endpoint_factor','selector','label_factor','record_order','correct_index','opposite_index','decoy_indices']);assert u['layer']==layer and set(raw[u['id']])=={'semantic','random'}
  for kind in ['semantic','random']:
   rec=u[kind];a=rec['audit'];assert a['actual_original_SDPA_invocations']==48 and a['explicit_FP64_attention_calls']==1 and a['prefix_identity_sites']==194 and a['unchanged_nonH_checks']==1 and a['receiver_row_writebacks']==q['R']+1-q['H_start'];assert len(a['events'])==1;e=a['events'][0];assert [e['loop'],e['layer']]==[2,layer];assert e['record_spans']==q['record_spans'] and e['receiver_span']==[q['H_start'],q['R']+1] and e['same_layer_current_QKV'] and e['all_heads'];assert e['FP64_native_reconstruction_relative_l2']<=.01;maxrecon=max(maxrecon,e['FP64_native_reconstruction_relative_l2']);events+=1
   for k in ['actual_delta_norms','projected_actual_delta_norms','requested_paired_delta_norms','native_norms']:assert len(e[k])==q['R']+1-q['H_start'] and np.isfinite(e[k]).all()
   assert rec['scores']['1']['full49152_sha256']==u['native']['scores']['1']['full49152_sha256']
   for loop in [1,2]:endpoint(raw[u['id']][kind][loop],rec['scores'][str(loop)],q)
  d=u['random']['audit']['events'][0]['dose'];s=u['semantic']['audit']['events'][0];assert d['target']==s['actual_delta_norms'];assert np.allclose(d['actual'],u['random']['audit']['events'][0]['actual_delta_norms'],atol=1e-11,rtol=1e-11);assert d['matched']==[abs(a-t)<=max(1e-9,.01*t) for a,t in zip(d['actual'],d['target'])];rows.append(u)
 print('AUDITED',p.stem,count,flush=True)
assert len(rows)==4096 and len({(u['id'],u['layer']) for u in rows})==4096
for layer in range(9,17):assert {u['id'] for u in rows if u['layer']==layer}==set(qs)
scientific=count;t=json.loads((R/'TECHNICAL_GPU.json').read_text());tr=torch.load(RAW/'technical_heads.pt',map_location='cpu',weights_only=False)
for c in t['cases']:
 q=qs[c['id']]
 for kind,rec in c['conditions'].items():
  for loop in [1,2]:endpoint(tr[c['id']][kind][loop],rec['scores'][str(loop)],q)
assert count-scientific==24 and scientific==16384
# Fraction arithmetic deliberately separate from NumPy selector.
def y(u,k):
 x=list(map(F,u[k]['scores']['2']['candidate_logits']));return x[u['opposite_index']]-(x[u['decoy_indices'][0]]+x[u['decoy_indices'][1]])/2
def mean(xs):return sum(xs,F(0))/len(xs)
sel=json.loads((R/'LAYER_SELECTION.json').read_text());mins={}
for l in range(9,17):
 rr=[u for u in rows if u['layer']==l];p=sel['candidate_profiles'][str(l)];four={f'F{f}_E{e}':mean([y(u,'semantic')-y(u,'random') for u in rr if u['firsthop_factor']==f and u['endpoint_factor']==e]) for f in [0,1] for e in [0,1]};fine={f'F{f}_E{e}_Q{s}':mean([y(u,'semantic')-y(u,'random') for u in rr if (u['firsthop_factor'],u['endpoint_factor'],u['selector'])==(f,e,s)]) for f in [0,1] for e in [0,1] for s in [0,1]};assert {k:float(v) for k,v in four.items()}==p['four_strata'];assert {k:float(v) for k,v in fine.items()}==p['eight_selector_strata'];assert float(mean([y(u,'semantic')-y(u,'native') for u in rr]))==p['mean_semantic_vs_native_Y'];mins[l]=min(four.values());assert float(mins[l])==p['m_minimum_stratum_effect'];assert p['all_eight_positive']==all(v>0 for v in fine.values())
 target=[];actual=[]
 for u in rr:
  d=u['random']['audit']['events'][0]['dose'];target+=d['target'];actual+=d['actual']
 target=np.array(target);actual=np.array(actual);err=abs(target-actual);matched=err<=np.maximum(1e-9,.01*target);er=float(err.sum()/target.sum());mass=float(target[matched].sum()/target.sum());qc=er<=.01 and mass>=.99;assert p['dose_QC']['pass']==qc and p['dose_QC']['matched']==int(matched.sum()) and p['dose_QC']['norm_error_ratio']==er and p['dose_QC']['matched_target_mass_fraction']==mass;assert p['eligible']==(mins[l]>F(1,10) and all(v>0 for v in fine.values()) and qc)
best=max(mins,key=lambda l:(mins[l],-l));assert sel['maximum_minimum_candidate']==best;assert sel['selected_layer']==(best if sel['candidate_profiles'][str(best)]['eligible'] else None)
out={'status':'PASS','freeze_sha256':fs,'world_layer_rows':len(rows),'scientific_body_forwards':8192,'scientific_full49152_saved_endpoints':scientific,'technical_saved_endpoints':count-scientific,'full_logprob_max_error':maxlp,'events_checked':events,'prefix_assertions_covered':events*194,'max_native_reconstruction_global_relative_l2':maxrecon,'selection_independent_Fraction_replay':'PASS','selected_layer':best,'native_baselines':'Reused parent frozen SHA-bound anchors, not fresh forwards','executor':'Sol CPU self-audit; not independent neural replication','utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};(R/'DISCOVERY_FULL_VECTOR_CPU_AUDIT.json').write_text(json.dumps(out,indent=2));print('FULL_AUDIT_PASS',count,best,flush=True)
