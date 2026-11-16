import pathlib,json,torch,hashlib,datetime
R=pathlib.Path('/root/ouro-temporal-bridge-sol-20261117');RAW=pathlib.Path('/root/autodl-tmp/ouro-temporal-bridge-sol-20261117/raw');torch.set_num_threads(4);data=torch.load(RAW/'TECHNICAL_PREFIX_WITNESSES.pt',map_location='cpu',weights_only=False);qs={q['id']:q for q in map(json.loads,(R/'INPUTS.jsonl').read_text().splitlines())};checks=0
for rev in [0,1]:
 for a,b in [('original','donor_edit'),('donor','original_edit')]:
  x=data[f'TB000_L{rev}_{a}'];y=data[f'TB000_L{rev}_{b}']
  for key in x['kv']:
   for aa,bb in zip(x['kv'][key],y['kv'][key]):assert torch.equal(aa.view(torch.uint8),bb.view(torch.uint8));checks+=1
for world in ['original','donor','original_edit','donor_edit']:
 a=data['TB000_L0_'+world];b=data['TB000_L1_'+world]
 for key in a['kv']:
  for x,y in zip(a['kv'][key],b['kv'][key]):assert torch.equal(x.view(torch.uint8),y.view(torch.uint8));checks+=1
q0=qs['TB000_L0_original'];q1=qs['TB000_L0_original_edit'];first=min(i for i,(x,y) in enumerate(zip(q0['ids'],q1['ids'])) if x!=y);a=data[q0['id']];b=data[q1['id']];profiles={}
for key,(k0,v0) in a['kv'].items():
 k1,v1=b['kv'][key];dk=(k1.double()-k0.double()).square().sum(dim=(1,3)).sqrt()[0];dv=(v1.double()-v0.double()).square().sum(dim=(1,3)).sqrt()[0];assert not bool((dk[:first]!=0).any()) and not bool((dv[:first]!=0).any());profiles[key]={'nonzero_K_positions':torch.nonzero(dk).flatten().tolist(),'nonzero_V_positions':torch.nonzero(dv).flatten().tolist(),'K_difference_norm_by_token':dk.tolist(),'V_difference_norm_by_token':dv.tolist()}
(R/'SOURCE_PREFIX_DIFFERENCE_PROFILE.json').write_text(json.dumps({'scope':'FixedtechnicalTB000only; observednumericaldifferenceextent,notsemanticprobeorn ewindependentvariableassumption','first_different_input_position':first,'source_end_b':q0['b'],'profiles':profiles},indent=2));out={'status':'PASS','fullprefix_future_invariance_tensor_comparisons':checks,'prefix_difference_never_precedes_changed_inputtoken':True,'raw_witness_sha256':hashlib.sha256((RAW/'TECHNICAL_PREFIX_WITNESSES.pt').read_bytes()).hexdigest(),'scope':'CPUactualpostRoPEprefixK/V underfuturemappingandoptionchanges atall24layers,bothpasses. Differencecanpropagateintolaterprefixpunctuation; fullSneededbydesign.','utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};(R/'PREFIX_WITNESS_CPU_AUDIT.json').write_text(json.dumps(out,indent=2));print('PREFIX_CPU_AUDIT_PASS',checks)
