"""Serial matched four-loop, one-full-head deployment timing; technical cohort only."""
import CATC_CRITICAL as C
from CATC_CRITICAL import torch
import pathlib,json,contextlib,time,statistics,datetime
R=C.ROOT;O=pathlib.Path('/root/ouro-catc-sol-20261116');s=json.loads((R/'SELECTED_METHOD.json').read_text());f=json.loads((O/'FREEZE.json').read_text());qs={q['id']:q for q in map(json.loads,(O/'INPUTS.jsonl').read_text().splitlines())};m=C.initialize();result={}
@torch.inference_mode()
def deploy(q,kind):
 ids=torch.tensor([q['ids']],device='cuda');context=contextlib.nullcontext() if kind=='native' else C.control(m,q,*s['coefficients'],kind=kind)
 with context:_,states,_=m.model(ids,use_cache=False,past_key_values=None)
 return m.lm_head(states[3])[:,q['R']]
for kind in ['native','structured','matched_random','matched_scaling']:
 times=[];mem=[];hashes=[]
 for id in f['technical_ids']:
  q={**qs[id]};old=json.loads((O/'results'/f'{id}.json').read_text());target=next(z for z in old['records'] if z['name']==s['original_record']);q['_dose_targets']=[v[3] for v in target['norms']];_,zz,_=C.forward(m,q,*(s['coefficients'] if kind!='native' else [0.,0.]),kind='structured' if kind=='native' else kind);ref=zz[4]
  for rep in range(4):
   torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.monotonic();z=deploy(q,kind);torch.cuda.synchronize();elapsed=time.monotonic()-t;assert torch.equal(z.cpu().view(torch.uint8),ref.view(torch.uint8))
   if rep:times.append(elapsed);mem.append(torch.cuda.max_memory_allocated())
  hashes.append({'id':id,'K4_full49152_sha256':C.sha(z)})
 result[kind]={'median_seconds':statistics.median(times),'measured_repeats':len(times),'peak_allocated_bytes':max(mem),'times':times,'endpoint_hashes':hashes}
for v in result.values():v['relative_vs_native']=v['median_seconds']/result['native']['median_seconds']
C.save('MATCHED_DEPLOYMENT_PROFILE.json',{'status':'PASS','methods':result,'scope':'Threefixedtechnicalquestions,onewarmup+3measuredrepeats,serial4passes+onefullshapedK4head. Matchedcontrols usecachedoutcome-freecompanionCATCdoses; standalonecontroltimers excludeacquiringthosestructureddoses. Noearlyexitclaim. Allfull49152K4endpoints byte-equal collector.','utc':datetime.datetime.now(datetime.timezone.utc).isoformat()});print('DEPLOYMENT_PROFILE_PASS')
