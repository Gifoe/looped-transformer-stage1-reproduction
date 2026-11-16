import CATC_IMPLEMENTATION as C
from CATC_IMPLEMENTATION import ROOT,torch
import json,time,statistics
f=json.loads((ROOT/'FREEZE.json').read_text());qs={q['id']:q for q in map(json.loads,(ROOT/'INPUTS.jsonl').read_text().splitlines())};m=C.initialize();out={}
with torch.inference_mode():
 for method in f['methods']:
  runs=[]
  for id in f['technical_ids']:
   q=qs[id];x=torch.tensor([q['ids']],device='cuda');stored=torch.load(ROOT/'results'/f'{id}.pt',weights_only=False)[method['name']][4]
   for rep in range(4):
    torch.cuda.synchronize();start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    with C.control(m,q,**{k:method[k] for k in ['av','ap','kind']}) as rec:_,states,_=m.model(x,use_cache=False,past_key_values=None)
    z=m.lm_head(states[3])[:,q['R']];torch.cuda.synchronize();duration=time.monotonic()-start;assert torch.equal(z.cpu().view(torch.uint8),stored.view(torch.uint8))
    if rep:runs.append({'seconds':duration,'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
  out[method['name']]={'n':len(runs),'median_seconds':statistics.median(x['seconds'] for x in runs),'max_memory_bytes':max(x['peak_allocated_bytes'] for x in runs),'runs':runs}
 base=out['T0']['median_seconds']
 for name,v in out.items():v['relative_latency_vs_native_K4']=v['median_seconds']/base
# Verify the official forcedK2/K4 interfaces with no CATC hook installed.
 for id in f['technical_ids']:
  q=qs[id];x=torch.tensor([q['ids']],device='cuda');stored=torch.load(ROOT/'results'/f'{id}.pt',weights_only=False)['T0']
  for d in (2,4):
   z=m(x,use_cache=False,exit_at_step=d-1).logits[:,q['R']].cpu();assert torch.equal(z.view(torch.uint8),stored[d].view(torch.uint8))
 C.save('OFFICIAL_K2_K4_ENDPOINT_AUDIT.json',{'status':'PASS','cases':3,'full49152_bitwise':True})
C.save('MATCHED_K4_INFERENCE_PROFILE.json',{'status':'PASS','scope':'Serial same-device after science: full4loops, exactlyonefull-shaped K4head for bothnative and treatments;3fixedtechnicalcases,1warmup+3measured percase; instrumentation overhead included; eachfullendpointmustmatchscience. MaincaptureT0has4heads but treatments2, so capturetimersare not matched online inference latency.','methods':out})
print('MATCHED_PROFILE_PASS')
