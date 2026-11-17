import model
import region_operator as O
from guard import verify,sha,exact
import json,sys,time,torch,datetime
R=model.R;fs=verify();assert json.loads((R/'TECHNICAL_GATE.json').read_text())['status']=='PASS' and json.loads((R/'DEVELOPMENT_FULL_VECTOR_CPU_AUDIT.json').read_text())['status']=='PASS';sel=json.loads((R/'REGION_SELECTION.json').read_text())['selected_mask'];assert sel in range(1,8);rank=int(sys.argv[1]);assert rank in range(3);qs=[q for q in map(json.loads,(R/'INPUTS.jsonl').read_text().splitlines()) if q['split']=='region_confirmation' and int(q['pair_id'][2:])%3==rank];m=model.initialize('confirm'+str(rank));start=time.monotonic();outdir=R/'confirmation_results';outdir.mkdir(exist_ok=True)
for pid in sorted({q['pair_id'] for q in qs}):
 path=outdir/(pid+'.json');rawpath=O.RAW/(pid+'_confirm.pt');group=[q for q in qs if q['pair_id']==pid]
 if path.exists():
  old=json.loads(path.read_text());assert old['freeze_sha256']==fs and old['selected_mask']==sel and sha(rawpath)==old['raw_sha256'];exact(old['rows'],[q['id'] for q in group]);continue
 rows=[];raw={};label={}
 for q in group:
  native,nh,pref,_=O.run(m,q,kind='native',capture_prefix=True);assert len(pref)==194;sem,sh,_,_=O.run(m,q,kind='semantic',mask=sel,native_prefix=pref);ran,rh,_,_=O.run(m,q,kind='random',mask=sel,native_prefix=pref,target_doses={'11':sem['audit']['events'][0]['actual_delta_norms']});raw[q['id']]={'native':{d:x.cpu() for d,x in nh.items()},'semantic':{d:x.cpu() for d,x in sh.items()},'random':{d:x.cpu() for d,x in rh.items()}}
  if sel==7:fullsem,fullran=sem,ran
  else:
   fullsem,fh,_,_=O.run(m,q,kind='semantic',mask=7,native_prefix=pref);fullran,gh,_,_=O.run(m,q,kind='random',mask=7,native_prefix=pref,target_doses={'11':fullsem['audit']['events'][0]['actual_delta_norms']});raw[q['id']].update({'full_semantic':{d:x.cpu() for d,x in fh.items()},'full_random':{d:x.cpu() for d,x in gh.items()}})
  for rec in [sem,ran,fullsem,fullran]:assert rec['scores']['1']==native['scores']['1']
  selected=[t for bit,span in zip([1,2,4],[q['Q_span'],q['O_span'],[q['R'],q['R']+1]]) if sel&bit for t in range(*span)];assert sem['audit']['events'][0]['actual_delta_norms']==[v if t in selected else 0. for t,v in zip(range(q['H_start'],q['R']+1),fullsem['audit']['events'][0]['actual_delta_norms'])]
  key=tuple(q[k] for k in ['firsthop_factor','endpoint_factor','selector','record_order'])
  for name,rec in [('native',native),('semantic',sem),('full_semantic',fullsem)]:
   hashes=rec['audit']['Q_prefix_hashes'];k=(name,key)
   if q['label_factor']:assert hashes==label[k],('freshlabelprefix',q['id'],name)
   else:label[k]=hashes
  meta={k:q[k] for k in ['id','pair_id','template','firsthop_factor','endpoint_factor','selector','label_factor','record_order','correct_index','opposite_index','decoy_indices','Q_span','O_span','H_start','R']};rows.append({**meta,'mask':sel,'native':native,'semantic':sem,'random':ran,'full_semantic':fullsem,'full_random':fullran,'canonical_native_prefix':pref})
 torch.save(raw,rawpath);path.write_text(json.dumps({'pair_id':pid,'selected_mask':sel,'freeze_sha256':fs,'raw_sha256':sha(rawpath),'rows':rows}));print('REGION_CONFIRM',rank,pid,round(time.monotonic()-start,1),flush=True)
(R/f'CONFIRM_SHARD_{rank}_COMPLETE.json').write_text(json.dumps({'status':'COMPLETE','rank':rank,'freeze_sha256':fs,'selected_mask':sel,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}))
