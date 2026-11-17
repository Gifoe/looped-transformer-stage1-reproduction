import pathlib,json,torch,numpy as np,hashlib,sys,datetime
from guard import verify,sha,exact
R=pathlib.Path('/root/ouro-region-use-sol-20261117');RAW=pathlib.Path('/root/autodl-tmp/ouro-region-use-sol-20261117/raw');torch.set_num_threads(4);fs=verify();phase=sys.argv[1];assert phase in ['development','confirmation'];qs={q['id']:q for q in map(json.loads,(R/'INPUTS.jsonl').read_text().splitlines()) if q['split']==('development' if phase=='development' else 'region_confirmation')};seen=[];count=0;events=0;maxlp=0.;maxrecon=0.;labels=0
def endpoint(x,s,q):
 global count,maxlp
 assert x.numel()==49152 and x.dtype==torch.bfloat16;assert hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()==s['full49152_sha256'];v=x.flatten().double().numpy();c=v[q['targets']];assert c.tolist()==s['candidate_logits'];ties=np.flatnonzero(c==c.max()).tolist();assert ties==s['ties'] and ties[0]==s['prediction'];m=v.max();err=float(abs(c-m-np.log(np.exp(v-m).sum())-s['candidate_logprobs']).max());assert err<2e-5;maxlp=max(maxlp,err);count+=1
folder=R/('results' if phase=='development' else 'confirmation_results')
for p in sorted(folder.glob('SR*.json')):
 z=json.loads(p.read_text());assert z['freeze_sha256']==fs;rr=z['rows'];exact(rr,[q['id'] for q in qs.values() if q['pair_id']==z['pair_id']]);rp=RAW/(p.stem+('.pt' if phase=='development' else '_confirm.pt'));assert sha(rp)==z['raw_sha256'];raw=torch.load(rp,map_location='cpu',weights_only=False);assert set(raw)=={u['id'] for u in rr};label={}
 for u in rr:
  q=qs[u['id']];mask=u['mask'];seen.append((u['id'],mask));assert all(u[k]==q[k] for k in ['pair_id','template','firsthop_factor','endpoint_factor','selector','label_factor','record_order','correct_index','opposite_index','decoy_indices','Q_span','O_span','H_start','R']);expected={'semantic','random'} if phase=='development' else ({'native','semantic','random'} if mask==7 else {'native','semantic','random','full_semantic','full_random'});assert set(raw[u['id']])==expected
  for name,hh in raw[u['id']].items():
   rec=u[name];a=rec['audit'];assert a['actual_original_SDPA_invocations']==48
   if name!='native':
    mm=7 if name.startswith('full_') else mask;assert a['explicit_FP64_attention_calls']==1 and a['prefix_identity_sites']==194 and a['unchanged_nonH_checks']==1 and len(a['events'])==1;e=a['events'][0];assert [e['loop'],e['layer']]==[2,11] and e['region_mask']==mm and e['closed_rows_unchanged'] and e['record_spans']==q['record_spans'] and e['receiver_span']==[q['H_start'],q['R']+1];opened=[t for bit,span in zip([1,2,4],[q['Q_span'],q['O_span'],[q['R'],q['R']+1]]) if mm&bit for t in range(*span)];assert opened==e['selected_rows'] and a['selected_row_count']==len(opened) and a['receiver_row_writebacks']==len(opened);assert all(v==0 for t,v in zip(range(q['H_start'],q['R']+1),e['actual_delta_norms']) if t not in opened);assert e['FP64_native_reconstruction_relative_l2']<=.01;maxrecon=max(maxrecon,e['FP64_native_reconstruction_relative_l2']);events+=1;assert rec['scores']['1']==u['native']['scores']['1']
    if 'random' in name:
     sem=u['full_semantic' if name=='full_random' else 'semantic'];d=e['dose'];assert d['target']==sem['audit']['events'][0]['actual_delta_norms'] and d['actual']==e['actual_delta_norms'];assert d['matched']==[abs(a-t)<=max(1e-9,.01*t) for a,t in zip(d['actual'],d['target'])]
   else:assert a['events']==[] and a['explicit_FP64_attention_calls']==0
   for loop in [1,2]:endpoint(hh[loop],rec['scores'][str(loop)],q)
   if name in ['native','semantic','full_semantic']:
    hashes=a['Q_prefix_hashes'];assert len(hashes)==(194 if name=='native' else 195);key=(name,tuple(q[k] for k in ['firsthop_factor','endpoint_factor','selector','record_order']))
    if q['label_factor']:assert hashes==label[key];labels+=1
    else:label[key]=hashes
 print('REGION_AUDITED',phase,p.stem,count,flush=True)
assert len(seen)==len(set(seen))==(3072 if phase=='development' else 2048);assert {x[0] for x in seen}==set(qs)
out={'status':'PASS','phase':phase,'freeze_sha256':fs,'world_condition_rows':len(seen),'saved_full49152_endpoints':count,'full_logprob_max_error':maxlp,'events_checked':events,'prefix_identity_assertions_covered':events*194,'actual_label_paired_Q_hash_comparisons':labels,'Q_hash_sites_each_semantic':195,'max_native_reconstruction_global_relative':maxrecon,'executor':'Sol fullvector CPU self-audit, not independent neural replication','utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};(R/(phase.upper()+'_FULL_VECTOR_CPU_AUDIT.json')).write_text(json.dumps(out,indent=2));print('REGION_FULL_CPU_AUDIT_PASS',phase,count,labels,flush=True)
