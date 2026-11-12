"""Three fixed-case technical capture of six missing old-corner suffix stages."""
import os,json,hashlib,time,fcntl,resource,torch
from pathlib import Path
from safetensors.torch import load_file,save_file
from transformers import AutoModelForCausalLM
import recovery_adapter,qkv_runner,technical,probe_eight
from e1_injection_observer import inject_sdpa_R,bits
from e1_donor_pool import OLD,sha
ROOT=Path('/home/luansiyu/loop-latent');B=ROOT/'repo/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112';P=B/'E1/technical_probe';OUT=B/'E1/six_corner_probe';SEL=['ACTAAP_2013_7_16','Mercury_405804','MSA_2012_8_30'];ANGLES=['G111','G112','G121','G211','G212','G221']
if not __debug__:raise RuntimeError('optimized Python forbidden')
def need(x,msg):
 if not x:raise ValueError(msg)
def desc(t):return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
@torch.inference_mode()
def one(model,ident,guard,split):
 t=time.monotonic();src,_,_=recovery_adapter.load(split,ident);e=src['natural'];R=e['R'];ids=torch.tensor([e['ids']],device='cuda');key=hashlib.sha256(ident.encode()).hexdigest();mp=OLD/split/(key+'.json');tp=mp.with_suffix('.safetensors');covpath=OLD/(split+'_independent_cuda_coverage.json'); expected={'train':'d257487676f31d0d122a913caad2c85369629ce1acca21bdabb5b2de0603c381','validation':'b4bf09b8a1d3c2f2c54740851e6482eb7a761e7091d70bff0d592d5f79df9f7c','test':'6af57d83f0c58c5832eb810c3a22bfbf3086db22f31bb37e41c3303fd5111526'};need(sha(covpath)==expected[split],'coverage');cov=json.loads(covpath.read_text());mch=[q for q in cov['questions'] if q['id']==ident];need(len(mch)==1 and sha(mp)==mch[0]['metadata_sha256'] and sha(tp)==mch[0]['tensor_sha256'],'source');old=json.loads(mp.read_text());tens=load_file(str(tp));core=model.model;prior=core.total_ut_steps;saved={};meta={'id':ident,'split':split,'R':R,'input_ids_sha256':old['input_ids_sha256'],'targets':e['targets'],'old_eight_metadata_sha256':sha(mp),'old_eight_tensor_sha256':sha(tp),'conditions':{}}
 try:
  _,bases=technical.verify(model,ids,R,e['targets'],e['states'],e['norms'],e['logits'],e['scores']);need(bits(bases[0],tens['G111_logits'].cuda()),'native baseline')
  for name in ANGLES:
   guard();core.total_ut_steps=1;value=tens[name+'_sdpa_R'].cuda();roots=[]
   def root(m,args,kw):
    x=args[0] if args else kw.get('input_ids');need(torch.is_tensor(x) and bits(x,ids) and kw.get('labels') is None and kw.get('use_cache') is False and kw.get('past_key_values') is None,'root');roots.append(1)
   h=model.register_forward_pre_hook(root,with_kwargs=True)
   try:
    with inject_sdpa_R(model,R,value,expected_x=src['states'][0]['x'].cuda()) as r:y=model(ids,labels=None,use_cache=False,exit_at_step=0)
   finally:h.remove()
   z=y.logits[0,R,:].unsqueeze(0);score=technical.score(z,e['targets']);need(len(roots)==1 and bits(z,tens[name+'_logits'].cuda()) and bits(r['stages']['attention_localnorm'],tens[name+'_a_R'].cuda()) and score==old['scores'][name] and qkv_runner.descriptors(r)==old['corner_receipts']['G111']['actual_receiver_qkv'] and r['outside_R_unchanged'],'six-corner parity')
   saved[name+'_logits']=z.cpu().contiguous().clone();saved[name+'_injected_sdpa_R']=value.cpu().contiguous().clone()
   for stage,x in r['stages'].items():saved[name+'_'+stage]=x.cpu().contiguous().clone()
   meta['conditions'][name]={'scores':score,'logits_sha256':desc(z),'actual_receiver_qkv':qkv_runner.descriptors(r),'outside_R_unchanged':True,'source_corner':name}
  guard();meta['question_wall_seconds']=time.monotonic()-t;return meta,saved
 finally:core.total_ut_steps=prior

