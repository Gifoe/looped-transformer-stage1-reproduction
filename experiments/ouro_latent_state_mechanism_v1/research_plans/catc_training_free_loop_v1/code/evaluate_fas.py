import CATC_ADAPTATION as C
from CATC_ADAPTATION import ROOT,torch
import json,pathlib,hashlib,time,datetime
R=ROOT.parent;spec=json.loads((ROOT/'FREEZE.json').read_text());deadline=datetime.datetime.fromisoformat(spec['budget_deadline_utc']).timestamp();selected=json.loads((ROOT/'catc14/COMPLETE.json').read_text());alpha=selected['alpha'];m=C.initialize();qs=list(map(json.loads,(ROOT/'FULL_ANSWER_INPUTS.jsonl').read_text().splitlines()));labels={q['id']:q['gold'] for q in map(json.loads,pathlib.Path('/root/ouro-computation-sol-20261116/prior/analysis_inputs/analysis_labels.jsonl').read_text().splitlines()) if q['split']=='validation'};out=ROOT/'fas_results';out.mkdir(exist_ok=True)
# AllmethodfactorsfrozenbeforeFASoutcomes; exactzerocoefficientsreuseidentity.
secondary=json.loads((ROOT/'VALIDATION_MODEL_SELECTION.json').read_text())['selected']['alpha'];methods={'native':[0.]*14,'fixedVplus':[.1,0.]*7,'catc14':alpha,'validation_selected_structured':secondary}
# FreshFASzero/nativeanchor atthreefixed hashordered questions, before scoringeffects.
anchors=[]
for q in sorted(qs,key=lambda q:hashlib.sha256(('CATC-FAS-tech|'+q['id']).encode()).hexdigest())[:3]:
 qq={**q,'targets':[330,389,340,422]};rr,z,_=C.forward(m,qq,coefficients=[0.]*14)
 with torch.inference_mode():
  inp=torch.tensor([q['ids']],device='cuda');official=m(inp,use_cache=False,exit_at_step=3).logits[:,q['R']].cpu();assert torch.equal(official.view(torch.uint8),z[4].view(torch.uint8))
 anchors.append(q['id'])
C.save('FAS_NATIVE_IDENTITY_PASS.json',{'status':'PASS','cases':anchors,'full49152_bitwise':True})
for index,q in enumerate(qs):
 p=out/f'{q["id"]}.json'
 if p.exists():continue
 sums={n:[] for n in ['nativeK2','nativeK4','LoopCD.25','LoopCD.5','LoopCD1','fixedVplusK4','catc14K4','validation_selected_structuredK4']};seconds=0.;physical=0;endpoint_hashes=[]
 for option,target in enumerate(q['answer_targets']):
  token_scores={k:[] for k in sums}
  for j,tid in enumerate(target):
   if time.time()>deadline:raise RuntimeError('Fivehourpilotdeadline:retaincompleteFASquestions')
   ids=q['ids']+target[:j];qq={'id':q['id']+f'_o{option}_t{j}','split':'validation','ids':ids,'R':len(ids)-1,'prefix_len':q['prefix_len'],'targets':[330,389,340,422],'input_ids_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest()};cache={}
   for name,a in methods.items():
    key=('zero',) if not any(a) else tuple(a)
    if key in cache:rr,z=cache[key]
    else:rr,z,_=C.forward(m,qq,coefficients=a);cache[key]=(rr,z);seconds+=rr['seconds'];physical+=1
    if name=='native':
     z1=z[1].cuda().float().reshape(-1);z2=z[2].cuda().float().reshape(-1);z4=z[4].cuda().float().reshape(-1);vals={'nativeK2':z2,'nativeK4':z4,'LoopCD.25':z4+.25*(z4-z1),'LoopCD.5':z4+.5*(z4-z1),'LoopCD1':z4+(z4-z1)}
    else:vals={name+'K4':z[4].cuda().float().reshape(-1)}
    for n,v in vals.items():token_scores[n].append(float(v.log_softmax(-1)[tid]))
   endpoint_hashes.append({'option':option,'token':j,'prefix_sha256':qq['input_ids_sha256'],'native_K4_full49152_sha256':rr['scores']['4']['logits_sha256'] if name=='native' else C.sha(cache[('zero',)][1][4])})
  for n,v in token_scores.items():
   total=float(torch.tensor(v,dtype=torch.float32,device='cuda').sum());sums[n].append({'sum':total,'per_token':total/len(v),'per_character':total/q['answer_char_lengths'][option],'token_logprobs':v})
 result={'id':q['id'],'gold':labels[q['id']],'methods':{},'seconds':seconds,'physical_forwards':physical,'answer_char_lengths':q['answer_char_lengths'],'target_lengths':[len(t) for t in q['answer_targets']],'prefix_endpoints':endpoint_hashes}
 for n,v in sums.items():
  scores=[x['per_character'] for x in v];ties=[i for i,x in enumerate(scores) if x==max(scores)];result['methods'][n]={'scores':scores,'prediction':ties[0],'ties':ties,'options':v}
 p.write_text(json.dumps(result));print('FAS',index+1,len(qs),'forward_seconds',round(seconds,3),flush=True)
C.save('FULL_ANSWER_COMPLETE.json',{'status':'COMPLETE','n':len(qs),'alpha':alpha,'zero_selected_native_reused':not any(alpha),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
