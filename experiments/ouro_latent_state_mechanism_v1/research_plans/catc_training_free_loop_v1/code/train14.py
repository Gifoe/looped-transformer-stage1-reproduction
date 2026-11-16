import CATC_ADAPTATION as C
from CATC_ADAPTATION import ROOT,torch
import pathlib,json,numpy as np,time,datetime
R=ROOT.parent;F=json.loads((ROOT/'FREEZE.json').read_text());deadline=datetime.datetime.fromisoformat(F['budget_deadline_utc']).timestamp();qs={q['id']:q for q in map(json.loads,(R/'INPUTS.jsonl').read_text().splitlines())};val=list(map(json.loads,(ROOT/'VALIDATION_INPUTS.jsonl').read_text().splitlines()));labels={(q['split'],q['id']):q['gold'] for q in map(json.loads,pathlib.Path('/root/ouro-computation-sol-20261116/prior/analysis_inputs/analysis_labels.jsonl').read_text().splitlines())};m=C.initialize()
def allowed():
 if time.time()>deadline:raise RuntimeError('Originalfivehourpilotdeadline reached; preservepartialwork')
def ce(r,g):
 x=np.array(r['scores']['4']['candidate_logits'],dtype=np.float64);a=x.max();return float(a+np.log(np.exp(x-a).sum())-x[g])
def forward(q,a,kind):
 allowed();r,z,_=C.forward(m,q,kind=kind,coefficients=np.asarray(a).tolist());return r
# Required new-interface identity before fitting or validation selection.
for id in json.loads((R/'FREEZE.json').read_text())['technical_ids']:
 q=qs[id];old=torch.load(R/'results'/f'{id}.pt',weights_only=False)
 for name,a in [('T0',[0.]*14),('T1_structured',[.1,0.]*7),('T5_structured',[.1,.1]*7)]:
  r,z,_=C.forward(m,q,coefficients=a);assert all(torch.equal(v.view(torch.uint8),old[name][d].view(torch.uint8)) for d,v in z.items())
C.save('NUMERICAL_GATE.json',{'status':'PASS','native_all4_and_T1_T5_full49152_exact':True,'cases':3})
cache=ROOT/'validation_native.json'
if cache.exists():native=json.loads(cache.read_text())
else:
 native=[]
 for i,q in enumerate(val):native.append({'id':q['id'],'gold':labels['validation',q['id']],'result':forward(q,np.zeros(14),'structured')})
 cache.write_text(json.dumps(native));print('VALIDATION_NATIVE299_CAPTURED',flush=True)
base_loss=np.mean([ce(x['result'],x['gold']) for x in native]);order=np.random.default_rng(20261116).permutation(sorted(qs));all_schemes=['catc14','unstructured14','value7','routing7']
for scheme in all_schemes:
 out=ROOT/scheme;out.mkdir(exist_ok=True)
 if (out/'COMPLETE.json').exists():continue
 a=np.zeros(14);active=np.ones(14,dtype=bool)
 if scheme=='value7':active[1::2]=False
 if scheme=='routing7':active[0::2]=False
 kind='unstructured14' if scheme=='unstructured14' else 'structured';rng=np.random.default_rng(20261116);bestloss=float(base_loss);beststep=0;besta=a.copy();bad=0;hist=[];evals=0;seen=[]
 for t,start in enumerate(range(0,len(order),32)):
  batch=[qs[id] for id in order[start:start+32]];c=.05/(t+1)**.101;eta=.05/(t+1)**.602;direction=rng.choice([-1.,1.],size=14);direction[~active]=0;plus=np.clip(a+c*direction,-.25,.25);minus=np.clip(a-c*direction,-.25,.25);lp=[];lm=[]
  for q in batch:
   rp=forward(q,plus,kind);rm=forward(q,minus,kind);lp.append(ce(rp,labels['train',q['id']]));lm.append(ce(rm,labels['train',q['id']]));evals+=2;seen.append(q['id'])
  p=float(np.mean(lp))+.01*float(plus@plus);n=float(np.mean(lm))+.01*float(minus@minus);grad=(p-n)/(2*c)*direction;a=np.clip(a-eta*grad,-.25,.25);a[~active]=0;entry={'step':t+1,'ids':[q['id'] for q in batch],'plus':plus.tolist(),'minus':minus.tolist(),'loss_plus':p,'loss_minus':n,'alpha_after':a.tolist(),'evals':evals}
  if t+1 in [12,24,35] or start+32>=len(order):
   rows=[]
   for q in val:rows.append({'id':q['id'],'gold':labels['validation',q['id']],'result':forward(q,a,kind)})
   loss=float(np.mean([ce(x['result'],x['gold']) for x in rows]));(out/f'validation_step{t+1}.json').write_text(json.dumps({'alpha':a.tolist(),'rows':rows,'candidateCE':loss}));entry['validation_CE']=loss
   if loss<bestloss:bestloss=loss;beststep=t+1;besta=a.copy();bad=0
   else:bad+=1
  hist.append(entry);(out/'history.json').write_text(json.dumps(hist));print(json.dumps({'scheme':scheme,'step':t+1,'train_evals':evals,'best_validation_step':beststep,'best_CE':bestloss}),flush=True)
  if bad>=2:break
 C.save(f'{scheme}/COMPLETE.json',{'scheme':scheme,'kind':kind,'parameters':int(active.sum()),'alpha':besta.tolist(),'best_step':beststep,'validation_CE':bestloss,'initial_validation_CE':float(base_loss),'training_forward_evaluations':evals,'training_distinct_ids':len(set(seen)),'steps':len(hist),'weights_updated':False,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
C.save('TRAINING_COMPLETE.json',{'status':'COMPLETE','schemes':all_schemes,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
