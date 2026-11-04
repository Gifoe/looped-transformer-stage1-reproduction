"""Single reviewed R23 fullV pilot; targets only used after direction construction."""
import os,time,json
import numpy as np
from r23_common import ROOT,LOCK,read,sha,digest,jsonlines,save,gate

def measure(model,scorer,tok,request,hidden,expected,native2,index,chars,permutation,check):
 import torch
 from runtime import encode_pair,check_budget
 from r7_inference import forward_states,original_sums,target_lp
 from r7_references import prefix_alignment
 from r7_math import token_logits
 from r23_readouts import heads
 from r23_math import torch_logits,numpy_evolution
 from r23_cpu_checks import explicit
 pair=encode_pair(tok,*request);assert hidden['candidate']==index and hidden['target_ids']==pair[1];q=heads(model,hidden['normalized'],[1,2,3,4,5]);original=original_sums({k:q[k] for k in [1,4,5]},pair[1]);assert all(v==expected['original_methods'][k]['sum_scores'][index] for k,v in original.items())
 assert float(target_lp(q[2],pair[1]).sum())==native2[index] and float(target_lp(q[3],pair[1]).sum())==expected['native3_sum_scores'][index]
 readout_checks=[]
 if check:
  for depth in [2,3]:
   independent=heads(model,forward_states(scorer,pair,depth),[depth])[depth];assert torch.equal(independent,q[depth]);readout_checks.append(dict(context='Q',depth=depth,fullV_exact=True))
 refs=[];maps=[];reference_sums=[];prompts=read('configs/r7_reference_manifest.json')['references']
 for j,key in enumerate(['N0','N1','N2']):
  check_budget(5);rp=encode_pair(tok,prompts[key],request[1]);mapping,reason=prefix_alignment(tok,request,pair,[rp]);assert mapping is not None,reason
  rz=heads(model,forward_states(scorer,rp,4),[1,2,3,4]);observed=original_sums({k:rz[k] for k in [1,4]},rp[1]);assert all(v==expected['prior_methods'][j][k]['sum_scores'][index] for k,v in observed.items())
  if check:
   for depth in [2,3]:
    independent=heads(model,forward_states(scorer,rp,depth),[depth])[depth];assert torch.equal(independent,rz[depth]);readout_checks.append(dict(context=key,depth=depth,fullV_exact=True))
  refs.append(rz);maps.append(mapping[0]);reference_sums.append(observed)
 names=['primary','M3','q_only','reference_only','shuffle','probability_contrast','leaveout_0','leaveout_1','leaveout_2','M2'];pieces={k:[] for k in names};packed={};cpu=None
 for begin in range(0,len(pair[1]),32):
  end=min(begin+32,len(pair[1]));qs=torch.stack([q[k][begin:end] for k in [1,2,3,4]]);rs=torch.stack([torch.stack([r[k][torch.as_tensor(m[begin:end],device=q[4].device)] for k in [1,2,3,4]]) for r,m in zip(refs,maps)])
  out,diag,parts=torch_logits(qs,rs,permutation);m2,md=token_logits(qs[0],qs[3],rs[:1,0],rs[:1,3]);out['M2']=m2;diag['M2_alpha']=md['alpha']
  if check and begin==0:
   uq,dq=numpy_evolution(qs[:,:1].cpu().numpy());ur=np.stack([numpy_evolution(r[:,:1].cpu().numpy())[0] for r in rs]);c=(uq[None]-ur).mean(0);ce=float(np.max(np.abs(c-parts[2][:1].cpu().numpy())));pe=float(np.max(np.abs(dq['latent']-parts[3][:1].cpu().numpy())));assert max(ce,pe)<=1e-10
   ez,ed=explicit(qs[:,:1,:13].cpu().numpy());uz,dz=numpy_evolution(qs[:,:1,:13].cpu().numpy());onehot=float(np.max(np.abs(ez-uz)));assert onehot<=1e-10
   reconstructed=(out['M3'][:1].double().cpu().numpy()+c).astype(np.float32);ze=float(np.max(np.abs(reconstructed-out['primary'][:1].cpu().numpy())));assert ze<=2e-5;cpu=dict(c_error=ce,P_latent_error=pe,FP32_logit_error=ze,explicit_small_onehot_error=onehot)
  for k,z in out.items():pieces[k].append(target_lp(z,pair[1][begin:end]))
  for k,v in diag.items():packed.setdefault(k,[]).extend(v.cpu().tolist())
 sums={k:float(torch.cat(v).sum()) for k,v in pieces.items()};late=expected['original_methods']['native_k4']['normalized_scores'][index]-expected['original_methods']['native_k5']['normalized_scores'][index];scores={k:v/int(chars)+late for k,v in sums.items()}
 for name,old in [('M2','M2_N0_token'),('M3','M3_multi_token')]:
  assert scores[name]==expected['method_scores'][old][index] and packed[name+'_alpha']==expected['token_diagnostics'][index][old]['alpha']
 return dict(choice=index,characters=int(chars),tokens=len(pair[1]),sums=sums,scores=scores,late=late,metrics=packed,baseline_parity_error=0,native2_native3_parity_error=0,readout_checks=readout_checks,CPU_numpy_check=cpu,reference_positions=maps,original_sums=original,reference_sums=reference_sums,request_sha256=digest(request),target_ids_sha256=digest(pair[1]))

def main():
 started=time.monotonic();spec=gate()
 from r19_common import inputs
 data=inputs(spec);authority={r['id']:r for split in ['discovery','selection'] for r in jsonlines('results/r7_'+split+'_development.jsonl')};native2={r['id']:r['original_methods']['native_k2']['sum_scores'] for split in ['discovery','selection'] for r in jsonlines('results/slcd_r2_'+split+'_development.jsonl')}
 from runtime import load_model,check_budget
 from ouro_cached_scoring import OuroCachedScorer
 from weight_edits import tensor_sha256
 import torch
 assert not (ROOT/'results/r23_pilot.jsonl').exists();t=time.monotonic();tok,model=load_model();scorer=OuroCachedScorer(model,5);load_seconds=time.monotonic()-t;identity=spec['model_identity'];assert sha('models/Ouro-1.4B/model.safetensors')==identity['safetensors_sha256']
 for p,h in identity['metadata_and_tokenizer_sha256'].items():assert sha('models/Ouro-1.4B/'+p)==h
 before={n:tensor_sha256(p) for n,p in model.named_parameters()};permutation=np.random.default_rng(2021).permutation(model.config.vocab_size);count=0;peak=0;torch.cuda.reset_peak_memory_stats()
 with (ROOT/'results/r23_pilot.jsonl').open('x',encoding='utf8') as out:
  for chosen,clean,expected,state in data:
   check_budget(5);assert time.monotonic()<float(os.environ['R23_SOFT_MONOTONIC']);payload=torch.load(ROOT/state['file'],map_location='cpu',weights_only=True);assert payload['id']==chosen['id'];expected['method_scores']=authority[chosen['id']]['method_scores']
   candidates=[measure(model,scorer,tok,req,payload['candidate_states'][j],expected,native2[chosen['id']],j,clean['choice_char_lengths'][j],permutation,count==0 and j==0) for j,req in enumerate(clean['request_pairs'])]
   from r7_inference import methods_from_sums
   from r7_core import baseline_vectors
   lengths=np.asarray(clean['choice_char_lengths']);orig=methods_from_sums({k:[c['original_sums'][k] for c in candidates] for k in candidates[0]['original_sums']},lengths);prior=methods_from_sums({k:[c['reference_sums'][0][k] for c in candidates] for k in candidates[0]['reference_sums'][0]},lengths);baselines=baseline_vectors(orig,prior);assert all(v.tolist()==authority[chosen['id']]['baseline_scores'][k] for k,v in baselines.items())
   out.write(json.dumps(dict(id=chosen['id'],split=chosen['split'],original_baseline_vectors_parity_error=0,candidates=candidates),allow_nan=False)+'\n');out.flush();os.fsync(out.fileno());count+=1;peak=max(peak,torch.cuda.max_memory_allocated());save('results/r23_PROGRESS.json',dict(completed=count,target=24,elapsed_seconds=time.monotonic()-started))
 after={n:tensor_sha256(p) for n,p in model.named_parameters()};assert before==after and count==24
 save('results/r23_COMPLETE.json',dict(status='complete',n=count,protocol_sha256=sha(LOCK),score_file_sha256=sha('results/r23_pilot.jsonl'),all_parameter_hashes_before=before,all_parameter_hashes_after=after,weights_modified=False,elapsed_seconds=time.monotonic()-started,model_load_seconds=load_seconds,peak_cuda_allocated_bytes=peak,memory_scope='GB10 unified allocator only',timing_scope='cachedmain exploratorypilot, not deployment throughput',heldout_accessed=False,KV_cache=False,explicit_attention_mask=False))
if __name__=='__main__':main()
