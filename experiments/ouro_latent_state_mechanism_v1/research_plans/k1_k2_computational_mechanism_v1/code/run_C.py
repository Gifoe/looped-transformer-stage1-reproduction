"""Bounded causal routing extension; no layer search or fitting."""
import core,json,sys,hashlib,time,datetime,contextlib
from core import ROOT,torch,forward,record,cpu,bits,tsave,jsave,_hidden,_replace
@contextlib.contextmanager
def routes(model,q,*,delta=None,permuted=False,norm=False,r_site=None,r_value=None):
 rec={'attention':{},'R_inputs':{},'dose':[]};handles=[];active=[];R=q['R'];g=torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(('C-dose-v1|'+q['id']).encode()).digest()[:8],'big')&((1<<63)-1));perm=torch.randperm(2048,generator=g).cuda()
 def pre(i):
  def fn(m,args,kw):
   active.append((i,kw['current_ut']))
   if kw['current_ut']==1 and i in (1,17):
    x=_hidden(args,kw);rec['R_inputs'][i]=x[:,R].detach().clone()
    if r_site==i:
     z=x.clone();z[:,R]=r_value;return _replace(args,kw,z)
  return fn
 def post(i):
  def fn(m,args,kw,out):assert active.pop()==(i,kw['current_ut'])
  return fn
 def a(i):
  def fn(m,args,out):
   if active[-1]!=(i,1):return
   before=out[:,R].detach().clone();rec['attention'][i]=before
   if delta is None:return
   d=delta[i].to(before.device)
   if permuted:d=d[:,perm]
   used=before.double()-d
   if norm:
    denom=used.norm();used=used*(before.double().norm()/denom) if float(denom)!=0 else used
   cast=used.to(out.dtype);z=out.clone();z[:,R]=cast;rec['dose'].append({'site':i+1,'desired_norm':float(d.norm()),'actual_norm':float((cast.double()-before.double()).norm()),'before_norm':float(before.double().norm()),'after_norm':float(cast.double().norm())});return z
  return fn
 try:
  for i,l in enumerate(model.model.layers):
   handles.extend([l.register_forward_pre_hook(pre(i),with_kwargs=True),l.register_forward_hook(post(i),with_kwargs=True)])
   if i in range(9,16):handles.append(l.input_layernorm_2.register_forward_hook(a(i)))
  yield rec
  assert set(rec['attention'])==set(range(9,16)) and set(rec['R_inputs'])=={1,17} and not active
 finally:
  for h in handles:h.remove()
def run(model,q,source,span,**kwargs):
 with routes(model,q,**kwargs) as a:
  z1,z,n=forward(model,q,mode='native' if source is None else 'boundary',donor=source,span=span)
 return z,n,a
phase=sys.argv[1];model,lock=core.initialize(worker='B');qs={q['id']:q for q in map(json.loads,(ROOT/'B_INPUTS.jsonl').read_text().splitlines())};allunits=[u for p in sorted((ROOT/'data/B').glob('*.json')) for u in json.loads(p.read_text())['units']];units=[u for u in allunits if u['pair_id']=='CW000'] if phase=='technical' else [u for u in allunits if u['split']=='heldout'];out=ROOT/'data/C';out.mkdir(exist_ok=True)
for idx,u in enumerate(units):
 path=out/f'{u["unit_id"]}.json'
 if path.exists():continue
 q=qs[u['receiver']];qd=qs[u['donor']];original=torch.load(ROOT/'data/B_native_restored'/f'{u["pair_id"]}.pt',weights_only=False);lo,hi=q['source_span'];source=original[qd['id']]['B'][:,lo:hi].cuda();expected={r['condition']:r for r in u['records']}
 zn,n,an=run(model,q,None,None);zp,p,ap=run(model,q,source,(lo,hi));assert core.digest(zn)==expected['native']['logits_sha256'] and core.digest(zp)==expected['source_boundary']['logits_sha256']
 delta={i:ap['attention'][i].double()-an['attention'][i].double() for i in range(9,16)};records=[];tensors={'N_attention':{i:cpu(x) for i,x in an['attention'].items()},'P_attention':{i:cpu(x) for i,x in ap['attention'].items()},'N_R_inputs':{i:cpu(x) for i,x in an['R_inputs'].items()},'P_R_inputs':{i:cpu(x) for i,x in ap['R_inputs'].items()},'N_logits':cpu(zn),'P_logits':cpu(zp)}
 conditions=[('restore_attention',{'delta':delta}),('attention_norm_control',{'delta':delta,'norm':True}),('attention_dose_control',{'delta':delta,'permuted':True}),('restore_early_R',{'r_site':1,'r_value':an['R_inputs'][1]}),('restore_L18_R',{'r_site':17,'r_value':an['R_inputs'][17]})]
 if phase=='technical':conditions.append(('zero_attention_identity',{'delta':{i:torch.zeros_like(v) for i,v in delta.items()}}))
 for condition,kw in conditions:
  z,n,a=run(model,q,source,(lo,hi),**kw)
  if condition=='zero_attention_identity':assert bits(z,zp)
  r=record(q,condition,z,n);r['route_dose']=a['dose'];records.append(r);tensors[condition]={'logits':cpu(z),'stages':{k:cpu(v) for k,v in n['stages'].items()}}
 tsave(path.with_suffix('.pt'),tensors);jsave(path,{'unit_id':u['unit_id'],'pair_id':u['pair_id'],'phase':phase,'predicted_cross_index':u['predicted_cross_index'],'original_index':u['original_index'],'donor_index':u['donor_index'],'source_reference':expected['source_boundary'],'native_reference':expected['native'],'records':records});print(json.dumps({'phase':phase,'done':idx+1,'total':len(units),'unit':u['unit_id']}),flush=True)
jsave(ROOT/('C_TECHNICAL_PASS.json' if phase=='technical' else 'C_COMPLETE.json'),{'status':'PASS','units':len(units),'native_and_source_full49152_reference_exact':True,'zero_attention_identity':phase=='technical','utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
