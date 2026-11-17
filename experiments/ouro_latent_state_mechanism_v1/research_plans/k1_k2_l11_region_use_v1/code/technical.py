import model
import region_operator as O
from guard import verify,sha
import json,torch,datetime
R=model.R;D=R.parent/'ouro-binding-interface-sol-20261117';P=model.P;fs=verify();qs=list(map(json.loads,(R/'INPUTS.jsonl').read_text().splitlines()));m=model.initialize('technical');raw={};cases=[];wits={};label={}
for pid in ['SR000','SR004','SR008']:
 for lab in [0,1]:
  q=next(q for q in qs if q['pair_id']==pid and q['firsthop_factor']==q['endpoint_factor']==q['selector']==q['record_order']==0 and q['label_factor']==lab)
  anchor=json.loads((P/'native_results'/(pid+'.json')).read_text());nd=next(u for u in anchor['rows'] if u['id']==q['id']);pref=anchor['canonical_native_prefixes'][nd['prefix_key']]['hashes'];prior=next(u for u in json.loads((D/'results'/(pid+'_L11.json')).read_text())['rows'] if u['id']==q['id']);conds={};raw[q['id']]={}
  for name,kind,mask in [('native','native',7),('mask0','semantic',0),('sem7','semantic',7),('rand7','random',7),('sem1','semantic',1),('rand1','random',1)]:
   target={str(11):conds['sem'+str(mask)]['audit']['events'][0]['actual_delta_norms']} if kind=='random' else None
   z,h,_,w=O.run(m,q,kind=kind,arm=2,native_prefix=pref,mask=mask,witness=True,target_doses=target);conds[name]=z;raw[q['id']][name]={d:x.cpu() for d,x in h.items()};wits[q['id']+'_'+name]={'q':q,'events':w,'record':z};assert z['scores']['1']==nd['native']['scores']['1']
   if name in ['native','mask0']:assert z['scores']==nd['native']['scores']
   if name in ['sem7','rand7']:assert z['scores']==prior['semantic' if name=='sem7' else 'random']['scores']
   if name=='sem1':
    ev=z['audit']['events'][0];full=conds['sem7']['audit']['events'][0];nq=q['Q_span'][1]-q['H_start'];assert ev['actual_delta_norms'][:nq]==full['actual_delta_norms'][:nq] and all(v==0 for v in ev['actual_delta_norms'][nq:])
   if kind in ['native','semantic']:
    hashes=z['audit']['Q_prefix_hashes'];key=(pid,name)
    if lab:assert hashes==label[key],('labelprefixnotidentical',key)
    else:label[key]=hashes
  cases.append({'id':q['id'],'conditions':conds});print('REGION_TECH_CASE',q['id'],flush=True)
torch.save(raw,O.RAW/'technical_heads.pt');torch.save(wits,O.RAW/'technical_witnesses.pt');(R/'TECHNICAL_GPU.json').write_text(json.dumps({'status':'GPU_PASS_PENDING_CPU','freeze_sha256':fs,'body_forwards':36,'cases':cases,'raw_heads_sha256':sha(O.RAW/'technical_heads.pt'),'witness_sha256':sha(O.RAW/'technical_witnesses.pt'),'actual_Q_labelprefix_bitidentity_cases':24,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2))
