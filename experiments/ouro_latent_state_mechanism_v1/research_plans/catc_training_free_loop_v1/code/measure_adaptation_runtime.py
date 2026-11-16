import CATC_ADAPTATION as C
from CATC_ADAPTATION import ROOT,torch
from predict_static import predict_static
import json,pathlib,statistics,time,datetime
R=ROOT.parent;m=C.initialize();qs={q['id']:q for q in map(json.loads,(R/'INPUTS.jsonl').read_text().splitlines())};cases=json.loads((R/'FREEZE.json').read_text())['technical_ids'];policies={'native':{'alpha':[0.]*14,'kind':'structured'},'fixedVplus':{'alpha':[.1,0.]*7,'kind':'structured'}}
for n in ['catc14','unstructured14','value7','routing7']:policies[n]=json.loads((ROOT/n/'COMPLETE.json').read_text())
policies['validation_selected_structured']=json.loads((ROOT/'VALIDATION_MODEL_SELECTION.json').read_text())['selected'];result={}
for name,v in policies.items():
 timings=[];peak=[]
 for id in cases:
  q=qs[id];_,z,_=C.forward(m,q,kind=v['kind'],coefficients=v['alpha']);ref=z[4]
  for rep in range(4):
   torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.monotonic();out=predict_static(m,q,v['alpha'],v['kind']);torch.cuda.synchronize();elapsed=time.monotonic()-t;assert torch.equal(out.cpu().view(torch.uint8),ref.view(torch.uint8))
   if rep:timings.append(elapsed);peak.append(torch.cuda.max_memory_allocated())
 result[name]={'median_seconds':statistics.median(timings),'repeats':len(timings),'peak_allocated_bytes':max(peak),'zero_native_bypass':not any(v['alpha']),'times':timings}
base=result['native']['median_seconds']
for v in result.values():v['relative_vs_native']=v['median_seconds']/base
C.save('SELECTED_INFERENCE_PROFILE.json',{'status':'PASS','scope':'SerialpostFAS,oneGPU,original4loopsandonefull-shapedK4head,3fixedtechnicalcasesx3measuredrepeatsafterwarmup. DeploymenthelperbypassesCATCwhenparametersareidenticallyzero. Everyfull49152endpointmustmatchcapturedmethod. Nonzeromethodsretainauditingoverhead.','methods':result,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()});print('SELECTED_INFERENCE_PROFILE_PASS')
