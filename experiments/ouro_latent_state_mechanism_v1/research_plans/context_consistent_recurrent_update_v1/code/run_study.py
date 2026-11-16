import INTERVENTION_IMPLEMENTATION as I
from INTERVENTION_IMPLEMENTATION import torch
import pathlib,json,datetime,time,numpy as np,hashlib
R=I.ROOT;f=json.loads((R/'FREEZE.json').read_text());deadline=datetime.datetime.fromisoformat(f['deadline_utc']).timestamp();assert json.loads((R/'TECHNICAL_PASS.json').read_text())['status']=='PASS';qs={q['id']:q for q in map(json.loads,(R/'INPUTS.jsonl').read_text().splitlines())};labels=json.loads((R/'LABELS.json').read_text());m=I.initialize();started=time.monotonic()
def collect(cohort,ids,conditions):
 for i,id in enumerate(ids):
  p=R/'results'/f'{id}.json'
  if p.exists():continue
  records={};obs={};native=None
  for name in conditions:
   if time.time()>deadline:raise RuntimeError('Hard180minute budget stop; retain completed rows')
   region,eta=f['conditions'][name];r,z,v=I.forward(m,qs[id],region,eta);records[name]=r;obs[name]=v
   if name=='N':native=v
   else:
    r['attention_change_vs_native']={k:{'delta_L2':float((x.double()-native[k].double()).norm()),'relative_delta':float((x.double()-native[k].double()).norm()/native[k].double().norm().clamp_min(1e-30)),'cosine':float((x.double()*native[k].double()).sum()/(x.double().norm()*native[k].double().norm()).clamp_min(1e-30))} for k,x in v.items()}
  torch.save(obs,R/'results'/f'{id}.pt');row={'id':id,'cohort':cohort,'gold':labels[id],'input_ids_sha256':qs[id]['input_ids_sha256'],'conditions':records};p.write_text(json.dumps(row))
  if (i+1)%20==0:print(cohort,i+1,len(ids),'wall_seconds',round(time.monotonic()-started,1),flush=True)
 (R/(cohort+'_COMPLETE.json')).write_text(json.dumps({'n':len(ids),'conditions':conditions,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}))
collect('development',f['development'],list(f['conditions']))
rows=[json.loads((R/'results'/f'{id}.json').read_text()) for id in f['development']]
def ce(v,g):
 x=np.array(v,dtype=np.float64);a=x.max();return float(a+np.log(np.exp(x-a).sum())-x[g])
scores={}
for n in ['Q+','Q-']:
 utility=[];loss=[]
 for q in rows:
  a=q['conditions'][n]['scores']['4'];b=q['conditions']['N']['scores']['4'];g=q['gold'];utility.append((int(a['prediction']==g)-int(b['prediction']==g))*int(len(a['ties'])==len(b['ties'])==1));loss.append(ce(a['candidate_logits'],g))
 scores[n]={'strict_net_correct':sum(utility),'mean_candidate_CE':float(np.mean(loss))}
choice=min(scores,key=lambda n:(-scores[n]['strict_net_correct'],scores[n]['mean_candidate_CE'],n!='Q+'));selected={'method':choice,'development':scores,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'criterion':f['selection']};(R/'SELECTED_METHOD.json').write_text(json.dumps(selected,indent=2));print('SIGN_FROZEN',choice,scores,flush=True);sign=choice[-1];collect('confirmation',f['confirmation'],['N',choice,'R'+sign,'P'+sign,'Random'+sign]);(R/'STUDY_COMPLETE.json').write_text(json.dumps({'status':'COMPLETE','wall_seconds':time.monotonic()-started,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}));print('STUDY_COMPLETE')
