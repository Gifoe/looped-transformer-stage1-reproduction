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
def one(model,ident,guard):
 t=time.monotonic();src,_,_=recovery_adapter.load('test',ident);e=src['natural'];R=e['R'];ids=torch.tensor([e['ids']],device='cuda');key=hashlib.sha256(ident.encode()).hexdigest();mp=OLD/'test'/(key+'.json');tp=mp.with_suffix('.safetensors');cov=json.loads((OLD/'test_independent_cuda_coverage.json').read_text());mch=[q for q in cov['questions'] if q['id']==ident];need(len(mch)==1 and sha(mp)==mch[0]['metadata_sha256'] and sha(tp)==mch[0]['tensor_sha256'],'source');old=json.loads(mp.read_text());tens=load_file(str(tp));core=model.model;prior=core.total_ut_steps;saved={};meta={'id':ident,'split':'test','R':R,'input_ids_sha256':old['input_ids_sha256'],'targets':e['targets'],'old_eight_metadata_sha256':sha(mp),'old_eight_tensor_sha256':sha(tp),'conditions':{}}
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

def main():
 lock=os.open(ROOT/'ouro_watch.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
 try:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);need(not OUT.exists(),'existing output');OUT.mkdir();pending=json.loads((P/'pending.json').read_text());replay=json.loads((P/'replay.json').read_text());need(replay['pending_sha256']==sha(P/'pending.json') and replay['status']=='E1_THREE_CASE_CUDA_SCORE_AND_ENDPOINT_REPLAY_PASSED' and all(sha(Path(k))==v for k,v in pending['source_hashes'].items()),'prior technical chain');files={str(Path(__file__).resolve()):sha(Path(__file__).resolve()),str(Path(__file__).with_name('e1_six_replay.py')):sha(Path(__file__).with_name('e1_six_replay.py')),str(P/'pending.json'):sha(P/'pending.json'),str(P/'replay.json'):sha(P/'replay.json')};recovery_adapter.initialize();start=time.monotonic()
  with torch.no_grad():model=AutoModelForCausalLM.from_pretrained(ROOT/'model',local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
  model.config.use_cache=False;model.model.config.use_cache=False;mh=probe_eight.model_hash(model);need(mh==pending['model_state_sha256'],'model');params=lambda:[(n,id(x),x.data_ptr(),x._version) for n,x in list(model.named_parameters())+list(model.named_buffers())];vs=params()
  def guard():need(params()==vs,'model mutation')
  for i,ident in enumerate(SEL):
   torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();m,t=one(model,ident,guard);torch.cuda.synchronize();m['peak_allocated_bytes']=torch.cuda.max_memory_allocated();m['peak_reserved_bytes']=torch.cuda.max_memory_reserved();tp=OUT/(str(i)+'.safetensors');save_file(t,str(tp));back=load_file(str(tp));need(set(back)==set(t) and all(torch.equal(back[k],t[k]) for k in back),'readback');m['tensor_sha256']=sha(tp);m['tensor_bytes']=tp.stat().st_size;probe_eight.write_json(OUT/(str(i)+'.json'),m);print('E1_SIX_CASE',i,ident,m['question_wall_seconds'],m['tensor_bytes'],flush=True)
  need(probe_eight.model_hash(model)==mh and all(sha(Path(k))==v for k,v in pending['source_hashes'].items()) and all(sha(Path(k))==v for k,v in files.items()),'final source');probe_eight.write_json(OUT/'pending.json',{'status':'E1_SIX_CORNER_PENDING_REPLAY','count':3,'prior_probe_pending_sha256':sha(P/'pending.json'),'prior_replay_sha256':sha(P/'replay.json'),'source_hashes':files,'model_state_sha256':mh,'runtime':pending['runtime'],'elapsed_seconds':time.monotonic()-start,'rss_peak_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'questions':[{'metadata_sha256':sha(OUT/(str(i)+'.json')),'tensor_sha256':sha(OUT/(str(i)+'.safetensors'))} for i in range(3)]});print('E1_SIX_PENDING_REPLAY',flush=True)
 finally:os.close(lock)
if __name__=='__main__':main()
