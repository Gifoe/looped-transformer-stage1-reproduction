"""Fixed technical-only numerical sensitivity; never select a scientific coefficient."""
import pathlib,importlib.util,json,hashlib,datetime
import CATC_IMPLEMENTATION as C
R=C.ROOT;torch=C.torch;f=json.loads((R/'FREEZE.json').read_text());qs={q['id']:q for q in map(json.loads,(R/'INPUTS.jsonl').read_text().splitlines())}
source=(R/'code/CATC_IMPLEMENTATION.py').read_text();old='qr=query[:,:,R:R+1,:].double();kk=key.double();vv=val[:,:,lo:hi].double()';new='qr=query[:,:,R:R+1,:].float();kk=key.float();vv=val[:,:,lo:hi].float()';assert old in source;source=source.replace(old,new);path=R/'technical/CATC_DECOMPOSITION_FP32.py';path.write_text(source)
spec=importlib.util.spec_from_file_location('catc_numeric_fp32',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);m=module.initialize();tests=[]
for id in f['technical_ids']:
 q=qs[id];saved=torch.load(R/'results'/f'{id}.pt',weights_only=False)
 for method in [x for x in f['methods'] if x['kind']=='structured']:
  rr,z,_=module.forward(m,q,method['av'],method['ap']);reference=saved[method['name']];depths=[]
  for d,v in z.items():
   ref=reference[d];equal=torch.equal(v.view(torch.uint8),ref.view(torch.uint8))
   if method['name']=='T0':assert equal
   a=C.score(v,q['targets']);b=C.score(ref,q['targets']);depths.append({'depth':d,'bitwise_equal':equal,'max_absolute_full_vocab_logit_change':float((v.float()-ref.float()).abs().max()),'prediction_changed':a['prediction']!=b['prediction'],'ties_changed':a['ties']!=b['ties'],'FP32_decomposition_candidates':a,'production_FP64_candidates':b})
  tests.append({'id':id,'method':method['name'],'depths':depths})
C.save('NUMERICAL_SENSITIVITY.json',{'status':'COMPLETE','scope':'3fixedtechnicalcases x7fixedglobalsettings; only decompositionQKV arithmeticFP64->FP32, original BF16backbone/SDPA and actual10%bound retained. No coefficient selection from this audit. Not fullcohort or hardware robustness.','zero_path_bitwise_exact':True,'tests':tests,'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()});print('NUMERICAL_SENSITIVITY_COMPLETE')
