"""32 pre-existing worlds; two complete map records. No new scientific outcomes."""
import record_operator as O
from record_operator import torch
import pathlib,json,datetime,hashlib,time
R=O.ROOT;P=pathlib.Path('/root/ouro-temporal-bridge-sol-20261117');qs=[q for q in map(json.loads,(P/'INPUTS.jsonl').read_text().splitlines()) if q['pair_id'] in ['TB000','TB001','TB008','TB016']];assert len(qs)==32;m=O.initialize();records=[];vectors={};witnesses={};eps=0.;joint_prediction_changes=[];self_count=0;start=time.monotonic()
for n,q in enumerate(qs):
 q=dict(q);lo,hi=q['regions']['map'];assert (hi-lo)%2==0;mid=(lo+hi)//2;q.update(record_spans=[[lo,mid],[mid,hi]],H_start=hi);assert q['ids'][lo:mid].count(198)==q['ids'][mid:hi].count(198)==1
 arm=1 if n%2==0 else 2;native,nh,pref,_=O.run(m,q,capture_prefix=True);group={'native':native};raw={'native':{k:v.cpu() for k,v in nh.items()}}
 for name in ['noop','self','K','V','joint']:
  z,heads,_,w=O.run(m,q,kind=name,arm=arm,native_prefix=pref,witness=n in [0,16,24]);group[name]=z;raw[name]={k:v.cpu() for k,v in heads.items()}
  if name in ['noop','self']:assert all(O.bits(nh[k],heads[k]) for k in [1,2]);self_count+=1
  if w:witnesses[q['id']+'_'+name]=w
 for left,right in [('joint','native'),('K','V')]:
  out={'left':left,'right':right,'full_logits_max_abs':{},'candidate_gap_max_abs':{},'prediction_or_tie_change':{}}
  for loop in [1,2]:
   x=raw[left][loop].double().flatten();y=raw[right][loop].double().flatten();target=q['targets'];off=(x[target]-y[target]);e=float((off[:,None]-off[None,:]).abs().max());out['full_logits_max_abs'][str(loop)]=float((x-y).abs().max());out['candidate_gap_max_abs'][str(loop)]=e;eps=max(e,eps);ls=group[left]['scores'][str(loop)];rs=group[right]['scores'][str(loop)];out['prediction_or_tie_change'][str(loop)]=ls['prediction']!=rs['prediction'] or ls['ties']!=rs['ties']
   if left=='joint' and len(rs['ties'])==1 and ls['prediction']!=rs['prediction']:joint_prediction_changes.append({'id':q['id'],'arm':arm,'loop':loop})
  group[left+'_vs_'+right]=out
 records.append({'id':q['id'],'arm':arm,'q':q,'conditions':group});vectors[q['id']]=raw;print('TECHNICAL_WORLD',n+1,len(qs),'epsilon_gap',eps,flush=True)
 if (n+1)%8==0:torch.save(vectors,O.RAW/('technical_heads_'+str(n//8)+'.pt'));vectors={}
torch.save(witnesses,O.RAW/'actual_SDPA_witnesses.pt');status='PASS' if not joint_prediction_changes else 'FAIL_PERMUTATION_NULL';out={'status':status,'worlds':32,'forwards':192,'source_population':'Old TB000/TB001/TB008/TB016 complete8world groups; arms alternate deterministicinputorder; no new semantic study data','noop_and_self_bitwise_cases':self_count,'epsilon_gap':eps,'practical_semantic_threshold':max(.1,10*eps),'joint_non_tied_prediction_changes':joint_prediction_changes,'records':records,'wall_seconds':time.monotonic()-start,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};(R/'TECHNICAL_NUMERICAL_GATE.json').write_text(json.dumps(out,indent=2));print('TECHNICAL_NUMERICAL_GATE',status,'epsilon',eps,'threshold',out['practical_semantic_threshold'])
