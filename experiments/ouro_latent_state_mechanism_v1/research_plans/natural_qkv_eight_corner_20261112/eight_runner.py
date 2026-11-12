"""Controlled same-question Q_R x KV run; GPU requires reviewed probe."""
import hashlib,json,torch
from pathlib import Path
from eight_observer import observe_qkv,bits
from natural_l18_hook import intervene
from natural_sources import sha,ROOT
import recovery_adapter
from technical import verify,score
from safetensors.torch import load_file
L18=ROOT/'research_plans/natural_l18_full_run_20261111';LCOV=ROOT/'research_plans/NATURAL_L18_INDEPENDENT_COVERAGE_20261111.json';LCOV_SHA='da4478aef58b320f5aa079f85b06ac70eab524922c340b2326a9487020da2535'
def require(x,s):
 if not x:raise ValueError(s)
def thash(t):return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
def describe(x):
 if torch.is_tensor(x):return {'shape':list(x.shape),'stride':list(x.stride()),'dtype':str(x.dtype),'device':str(x.device),'sha256':thash(x)}
 if x is None or isinstance(x,(str,int,float,bool)):return x
 raise ValueError('unserializable SDPA parameter')
def descriptors(record):return {'qkv':{k:describe(record[k]) for k in ('q','k','v')},'tail':[describe(v) for v in record['tail']],'kwargs':{k:describe(v) for k,v in record['kwargs'].items()}}
def snapshot(record):return {'q':record['q'].clone(),'k':record['k'].clone(),'v':record['v'].clone(),'tail':tuple(x.clone() if torch.is_tensor(x) else x for x in record['tail']),'kwargs':{k:(v.clone() if torch.is_tensor(v) else v) for k,v in record['kwargs'].items()}}
@torch.inference_mode()
def question(model,split,ident,guard):
 guard();src,old_m,old_path=recovery_adapter.load(split,ident);old_t=load_file(str(old_path));e=src['natural'];device=next(model.parameters()).device;require(device.type=='cuda' and not model.training and model.config._attn_implementation=='sdpa','model')
 runtime={'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(device)};require(all(e['runtime'][k]==v for k,v in runtime.items()) and e['runtime']['attention']=='sdpa' and e['runtime']['dtype']=='bfloat16' and e['runtime']['use_cache'] is False,'runtime')
 require(sha(LCOV)==LCOV_SHA,'L18 coverage SHA');coverage=json.loads(LCOV.read_text());require(coverage['status']=='INDEPENDENT_L18_CUDA_AND_SOURCE_COVERAGE_PASSED' and coverage['count']==2590,'L18 coverage');matches=[x for x in coverage['questions'] if x['id']==ident and x['split']==split];require(len(matches)==1,'L18 ID')
 key=split+'_'+hashlib.sha256(ident.encode()).hexdigest();mp=L18/(key+'.json');tp=mp.with_suffix('.safetensors');require(sha(mp)==matches[0]['metadata_sha256'] and sha(tp)==matches[0]['tensor_sha256'],'L18 source SHA');lm=json.loads(mp.read_text());lt=load_file(str(tp));require(lm['id']==ident and lm['R']==e['R'] and lm['targets']==e['targets'] and lm['gold']==e['gold'] and lm['natural_metadata_sha256']==e['metadata_sha256'] and lm['source_metadata_sha256']==src['source_metadata_sha256'],'L18 identity')
 ids=torch.tensor([e['ids']],device=device,dtype=torch.long);R=e['R'];targets=e['targets'];core=model.model;old=core.total_ut_steps;inputs_hash=hashlib.sha256(json.dumps(e['ids']).encode()).hexdigest()
 try:
  obs,base=verify(model,ids,R,targets,e['states'],e['norms'],e['logits'],e['scores']);guard()
  require(bits(base[0],lt['B1'].to(device)) and bits(base[1],lt['B2'].to(device)) and lm['scores']['B1']==e['scores'][0] and lm['scores']['B2']==e['scores'][1],'baseline source')
  captures={};capture_receipts={}
  for depth in (1,2):
   guard();core.total_ut_steps=depth;root_events=[]
   def root(m,args,kw):
    actual=args[0] if args else kw.get('input_ids');require(torch.is_tensor(actual) and bits(actual,ids) and kw.get('labels') is None and kw.get('use_cache') is False and kw.get('past_key_values') is None,'root input/mode');root_events.append(1)
   root_handle=model.register_forward_pre_hook(root,with_kwargs=True)
   try:
    own=src['states'][depth-1]
    with observe_qkv(model,depth,R) as qrec,intervene(model,depth,R,own['x'],own['a']) as arec:
     y=model(ids,labels=None,use_cache=False,exit_at_step=depth-1)
   finally:root_handle.remove()
   z=y.logits[0,R,:].unsqueeze(0);require(len(root_events)==1 and torch.isfinite(z).all() and bits(z,base[depth-1]) and score(z,targets)==e['scores'][depth-1] and bits(qrec['receiver_x'],own['x'].to(device)) and bits(arec['receiver_a'],own['a'].to(device)),'natural capture bridge')
   frozen=snapshot(qrec);desc=descriptors(frozen);captures[depth]=frozen;capture_receipts[str(depth)]={'status':'SEALED_COMPLETE_NATURAL_FORWARD','id':ident,'split':split,'input_ids_sha256':inputs_hash,'R':R,'depth':depth,'ut':depth-1,'physical_layer':17,'site':'post-RoPE SDPA actual entry','qkv':desc,'baseline_logits_sha256':thash(z),'x_R_sha256':thash(qrec['receiver_x']),'a_R_sha256':thash(arec['receiver_a']),'source_metadata_sha256':src['source_metadata_sha256'],'source_tensor_sha256':src['source_tensor_sha256']};require(capture_receipts[str(depth)]==old_m['capture_receipts'][str(depth)],'old capture descriptors');guard()
  donor=captures[2];donor_desc=descriptors(donor);receiver_desc=descriptors(captures[1]);results={};injections={};a_states={};scores={};corner_receipts={}
  for name,q2,k2,v2 in [('G111',False,False,False),('G112',False,False,True),('G121',False,True,False),('G122',False,True,True),('G211',True,False,False),('G212',True,False,True),('G221',True,True,False),('G222',True,True,True)]:
   guard();require(descriptors(donor)==donor_desc and descriptors(captures[1])==receiver_desc,'QKV mutation');core.total_ut_steps=1;arec=[];root_events=[]
   def root_angle(m,args,kw):
    actual=args[0] if args else kw.get('input_ids');require(torch.is_tensor(actual) and bits(actual,ids) and kw.get('labels') is None and kw.get('use_cache') is False and kw.get('past_key_values') is None,'angle root input/mode');root_events.append(1)
   def anorm(m,args,out):
    require(torch.is_tensor(out) and out.ndim==3 and torch.isfinite(out[:,R,:]).all(),'aR');arec.append(out[:,R,:].detach().clone())
   h=model.model.layers[17].input_layernorm_2.register_forward_hook(anorm);rh=model.register_forward_pre_hook(root_angle,with_kwargs=True)
   try:
    with observe_qkv(model,1,R,donor,q2,k2,v2) as rec:y=model(ids,labels=None,use_cache=False,exit_at_step=0)
   finally:h.remove();rh.remove()
   z=y.logits[0,R,:].unsqueeze(0);require(len(arec)==1 and len(root_events)==1 and descriptors(rec)==receiver_desc and torch.isfinite(z).all() and z.dtype==torch.bfloat16 and z.shape==(1,49152) and bits(rec['receiver_x'],src['states'][0]['x'].to(device)) and rec['outside_R_unchanged'] is True and descriptors(donor)==donor_desc,'receiver/output')
   results[name]=z.detach().clone();injections[name]=rec['injected_sdpa_R'].detach().clone();a_states[name]=arec[0];scores[name]=score(z,targets);corner_receipts[name]={'actual_receiver_qkv':descriptors(rec),'receiver_x_R_sha256':thash(rec['receiver_x']),'injected_sdpa_R_sha256':thash(injections[name]),'a_R_sha256':thash(a_states[name]),'outside_R_unchanged':rec['outside_R_unchanged'],'root_input_events':len(root_events)}
   if name=='G111':require(bits(z,base[0]) and bits(arec[0],src['states'][0]['a'].to(device)),'F11 endpoint')
   if name=='G222':require(bits(z,lt['T12'].to(device)) and bits(arec[0],src['states'][1]['a'].to(device)) and scores[name]==lm['scores']['T12'],'F22 endpoint')
  for new,old_name in [('G111','F11'),('G211','F21'),('G122','F12'),('G222','F22')]:
   require(bits(results[new].cpu(),old_t[old_name+'_logits']) and bits(injections[new].cpu(),old_t[old_name+'_sdpa_R']) and bits(a_states[new].cpu(),old_t[old_name+'_a_R']) and scores[new]==old_m['scores'][old_name],'old corner full parity')
  guard();return {'id':ident,'split':split,'R':R,'input_ids_sha256':inputs_hash,'targets':targets,'gold':e['gold'],'fixed_wrong':lm['fixed_wrong'],'natural_metadata_sha256':e['metadata_sha256'],'natural_tensor_sha256':e['tensor_sha256'],'branch_metadata_sha256':src['source_metadata_sha256'],'branch_tensor_sha256':src['source_tensor_sha256'],'l18_metadata_sha256':sha(mp),'l18_tensor_sha256':sha(tp),'l18_coverage_sha256':LCOV_SHA,'old_qkv_metadata_sha256':sha(old_path.with_suffix('.json')),'old_qkv_tensor_sha256':sha(old_path),'natural_tensor_scope':'historical_not_reread','natural_metadata_source':e['metadata_source'],'actual_branch_metadata_sha256':e['actual_branch_metadata_sha256'],'actual_branch_tensor_sha256':e['actual_branch_tensor_sha256'],'capture_receipts':capture_receipts,'corner_receipts':corner_receipts,'scores':scores,'logits':results,'injections':injections,'a_states':a_states}
 finally:core.total_ut_steps=old
