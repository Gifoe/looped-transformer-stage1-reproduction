"""Independent remainder1055 CUDA scoring / source / control audit."""
import os,json,hashlib,fcntl,torch
from pathlib import Path
from safetensors.torch import load_file
import e1_donor_pool
from e1_control_core import controls
from e1_donor_pool import OLD,sha
ROOT=Path('/home/luansiyu/loop-latent');B=ROOT/'repo/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112';E=B/'E1';OUT=E/'train';SEL=B/'orders/E1_DISCOVERY64_FROZEN.json';N=['G111','G112','G121','G211','G212','G221'];STAGES={'oproj','attention_localnorm','residual_after_attention','mlp_raw','mlp_localnorm','loop_postnorm'}|{f'layer{i+1:02d}_output' for i in range(17,24)}
if not __debug__:raise RuntimeError('optimized Python forbidden')
def need(x,s):
 if not x:raise ValueError(s)
def score(z,targets):
 lp=z.cuda().float().log_softmax(-1);return [float(lp.gather(-1,torch.tensor([[j]],device='cuda')).sum()) for j in targets]
def main():
 lock=os.open(ROOT/'ouro_watch.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
 try:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);rp=OUT/'remainder_receipt.json';r=json.loads(rp.read_text());selection=json.loads(SEL.read_text());priorids=[v['id'] for group in ('rescue','damage','stable_correct','stable_wrong') for v in selection['groups'][group]];d=json.loads((E/'discovery64_cuda_audit.json').read_text());oldcohort=json.loads((OLD/'train_independent_cuda_coverage.json').read_text());ids=[q['id'] for q in r['questions']];need(r['status']=='TRAIN_REMAINDER_COLLECTED_PENDING_INDEPENDENT_CUDA_AUDIT' and r['count']==len(ids)==1055 and len(set(ids))==1055 and d['status']=='DISCOVERY64_INDEPENDENT_CUDA_SCORE_AND_SOURCE_AUDIT_PASSED' and d['count']==64 and {q['id'] for q in d['questions']}==set(priorids) and set(ids)=={q['id'] for q in oldcohort['questions']}-set(priorids) and r['selection_sha256']==sha(SEL)=='6faad196db958744780ee6477af8839c693798b8b24d2a01a2d450c856c720e1' and all(sha(Path(k))==v for k,v in r['sources'].items()),'receipt/source');need(torch.cuda.is_available() and torch.__version__==r['runtime']['torch'] and torch.version.cuda==r['runtime']['cuda'] and torch.cuda.get_device_name()==r['runtime']['gpu'],'runtime');pool=e1_donor_pool.Pool();ledger={q['id']:q for q in r['questions']};need(set(ledger)==set(ids),'ID ledger');oldcov=json.loads((OLD/'train_independent_cuda_coverage.json').read_text());prior={q['id']:q for q in oldcov['questions']};out=[]
  for ident in ids:
   key=hashlib.sha256(ident.encode()).hexdigest();mp=OUT/(key+'.json');tp=mp.with_suffix('.safetensors');need(sha(mp)==ledger[ident]['metadata_sha256'] and sha(tp)==ledger[ident]['tensor_sha256'],'shard sha');m=json.loads(mp.read_text());t=load_file(str(tp));om=OLD/'train'/(key+'.json');op=om.with_suffix('.safetensors');need(m['id']==ident and m['split']=='train' and m['source_hashes']==r['sources'] and sha(om)==prior[ident]['metadata_sha256'] and sha(op)==prior[ident]['tensor_sha256'],'old SHA');old=json.loads(om.read_text());ot=load_file(str(op));need(m['input_ids_sha256']==old['input_ids_sha256'] and m['targets']==old['targets'] and m['R']==old['R'],'identity')
   for q in (1,2):
    cm=m['controls']['conditions'][str(q)];ss=[ot[f'G{q}{j}{k}_sdpa_R'] for j,k in ((1,1),(1,2),(2,1),(2,2))]
    for a in ('A1','A2'):
     z=t[f'q{q}_{a}_fp64'];need(z.dtype==torch.float64 and z.ndim==3 and z.shape[0]==1 and z.shape[1]==ss[0].shape[1] and z.shape[-1]>m['R'] and torch.isfinite(z).all() and (z>=0).all() and torch.allclose(z.sum(-1),torch.ones_like(z.sum(-1)),atol=1e-12,rtol=1e-12) and torch.count_nonzero(z[...,m['R']+1:])==0,'attention row')
    components=[t[f'q{q}_{k}_fp64'] for k in ('key','value','interaction','accounting_residual')];need(all(z.dtype==torch.float64 and z.shape==ss[0].shape and torch.isfinite(z).all() for z in components),'FP64 components');recon=[]
    for n,(j,k) in enumerate(((1,1),(1,2),(2,1),(2,2))):
     residual=t[f'q{q}_native_minus_ideal_{j}{k}'];need(residual.dtype==torch.float64 and residual.shape==ss[n].shape and torch.isfinite(residual).all(),'ideal residual');recon.append(ss[n].double()-residual)
    need(torch.allclose(recon[3]-recon[0],sum(components),atol=1e-10,rtol=1e-10) and torch.allclose((ss[3].double()-ss[2].double()-ss[1].double()+ss[0].double())-components[2],t[f'q{q}_native_minus_ideal_interaction_fp64'],atol=1e-10,rtol=1e-10),'FP64 accounting')
    first=controls(*ss,torch.ones_like(ss[3],dtype=torch.float64));donor=None
    if first['delta_norm']!=0:donor,dm=pool.select('train',ident,m['input_ids_sha256'],q);need(dm==cm['donor'],'donor provenance')
    C=controls(*ss,donor)
    for k in ('ADD_exact_fp64','cast_residual_fp64','delta_fp64','restore'):need(torch.equal(C[k],t[f'q{q}_{k}']),'arithmetic')
    for k in ('ID','ADD','ADD_NORM','UNRELATED'):
     if C[k] is None:need(k=='ADD_NORM' and cm['ADD_NORM_status']=='UNDEFINED','undefined');continue
     prefix=f'q{q}_{k}';z=t[prefix+'_logits'];need(z.dtype==torch.bfloat16 and tuple(z.shape)==(1,49152) and torch.isfinite(z).all() and torch.equal(t[prefix+'_injected_sdpa_R'],C[k]) and score(z,m['targets'])==cm[k]['scores'] and cm[k]['outside_R_unchanged'] and cm[k]['root_input_events']==1 and cm[k]['actual_receiver_qkv']==old['corner_receipts']['G111']['actual_receiver_qkv'],'control')
     for stage in STAGES:
      a=t[prefix+'_'+stage];need(a.dtype==torch.bfloat16 and tuple(a.shape)==(1,2048) and torch.isfinite(a).all(),'stage')
     if k=='ID':need(torch.equal(z,ot[f'G{q}22_logits']) and torch.equal(t[prefix+'_attention_localnorm'],ot[f'G{q}22_a_R']),'ID old')
   for n in N:
    z=t['old_'+n+'_logits'];need(torch.equal(z,ot[n+'_logits']) and torch.equal(t['old_'+n+'_attention_localnorm'],ot[n+'_a_R']) and torch.equal(t['old_'+n+'_injected_sdpa_R'],ot[n+'_sdpa_R']) and score(z,m['targets'])==m['old_stages']['conditions'][n]['scores'],'old stage')
    for stage in STAGES:
     a=t['old_'+n+'_'+stage];need(a.dtype==torch.bfloat16 and tuple(a.shape)==(1,2048) and torch.isfinite(a).all(),'old stage shape')
   out.append({'id':ident,'metadata_sha256':sha(mp),'tensor_sha256':sha(tp),'old_metadata_sha256':sha(om),'old_tensor_sha256':sha(op)})
   if len(out)%16==0:print('E1_REMAINDER_AUDITED',len(out),flush=True)
  need(all(sha(Path(k))==v for k,v in r['sources'].items()),'source drift');result={'status':'TRAIN_REMAINDER_INDEPENDENT_CUDA_SCORE_AND_SOURCE_AUDIT_PASSED','count':len(out),'selection_sha256':sha(SEL),'collector_receipt_sha256':sha(rp),'script_sha256':sha(Path(__file__)),'questions':out,'scope':'Independently replays CUDA scores and validates saved old endpoints, controls, stages, source; whole K/V were streamed, not archived; FP64 QKV reconstruction relies on collector and frozen source.'};p=E/'remainder_cuda_audit.json'
  with p.open('x') as f:json.dump(result,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
  print('E1_REMAINDER_AUDIT_COMPLETE',sha(p),flush=True)
 finally:os.close(lock)
if __name__=='__main__':main()
