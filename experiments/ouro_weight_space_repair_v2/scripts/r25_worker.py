"""R25 one bounded native-fullV Hellinger memory pilot; no learned weights."""
import os,time,json
import numpy as np
from r25_common import ROOT,LOCK,read,sha,digest,jsonlines,save,gate

def measure(model,scorer,tok,request,hidden,expected,index,chars,keys,key_logits,memory,key_input_ids,timing,check):
 import torch
 from runtime import encode_pair,check_budget
 from r7_inference import head_logits,forward_states,original_sums,target_lp
 from r7_references import prefix_alignment
 from r7_math import token_logits
 from r25_math import torch_retrieve,numpy_retrieve,torch_logits
 pair=encode_pair(tok,*request);assert hidden['candidate']==index and hidden['target_ids']==pair[1];maxvalue=max(e['source_token_position'] for e in memory['entries']);assert pair[0][:maxvalue+2]==key_input_ids[:maxvalue+2]
 def timed(name,fn):
  torch.cuda.synchronize();t=time.monotonic();v=fn();torch.cuda.synchronize();timing[name]+=time.monotonic()-t;return v
 q=timed('Q_head',lambda:head_logits(model,hidden['normalized']));original=original_sums(q,pair[1]);assert all(v==expected['original_methods'][k]['sum_scores'][index] for k,v in original.items());parity=[]
 if check:
  fresh=timed('checks',lambda:head_logits(model,forward_states(scorer,pair,4)));assert torch.equal(fresh[4],q[4]);parity.append(dict(context='Q',fresh_native4_vs_cached_fullV_exact=True))
 refs=[];maps=[];reference_sums=[];prompts=read('configs/r7_reference_manifest.json')['references']
 for j,name in enumerate(['N0','N1','N2']):
  check_budget(5);rp=encode_pair(tok,prompts[name],request[1]);mapping,reason=prefix_alignment(tok,request,pair,[rp]);assert mapping is not None,reason
  states=timed('N_forward',lambda:forward_states(scorer,rp,4));r=timed('N_head',lambda:head_logits(model,states));obs=original_sums(r,rp[1]);assert all(v==expected['prior_methods'][j][k]['sum_scores'][index] for k,v in obs.items())
  if check:
   fresh=timed('checks',lambda:head_logits(model,forward_states(scorer,rp,4)));assert torch.equal(fresh[4],r[4]);parity.append(dict(context=name,fresh_native4_vs_first_fullV_exact=True))
  refs.append(r);maps.append(mapping[0]);reference_sums.append(obs)
 aligned=[q[4]]+[r[4][torch.as_tensor(m,device=q[4].device)] for r,m in zip(refs,maps)];neighbors=[];retrieval=[]
 for j,z in enumerate(aligned):
  ni,dist,gap=timed('retrieval',lambda:torch_retrieve(keys,z));record=dict(context=['Q','N0','N1','N2'][j],neighbor_occurrences=ni.cpu().tolist(),minimum_two_distance_gap=gap.cpu().tolist())
  if check:
   begin=time.monotonic();ind,nd=numpy_retrieve(key_logits.cpu().numpy(),z[:1].cpu().numpy());error=float(np.max(np.abs(nd-dist[:1].cpu().numpy())));assert error<=1e-12 and int(ind[0])==int(ni[0]);record['first_numpy_check']=dict(all_distance_max_error=error,neighbor_exact=True,distances_Torch=dist[0].cpu().tolist(),distances_NumPy=nd[0].tolist());timing['checks']+=time.monotonic()-begin
  neighbors.append(ni);retrieval.append(record)
 neighbors=torch.stack(neighbors);names=['primary','M3','q_only','reference_only','memory_direction_only','frequency','raw_occurrence_shuffle','freq_bijection_matched','leaveout_0','leaveout_1','leaveout_2','M2'];parts={k:[] for k in names};packed={}
 for begin in range(0,len(pair[1]),32):
  end=min(begin+32,len(pair[1]));q1=q[1][begin:end];q4=q[4][begin:end];r1=torch.stack([r[1][torch.as_tensor(m[begin:end],device=q4.device)] for r,m in zip(refs,maps)]);r4=torch.stack([z[begin:end] for z in aligned[1:]])
  out,diag=timed('retrieval',lambda:torch_logits(q1,q4,r1,r4,neighbors[:,begin:end],memory['values'],memory['nulls']['raw_occurrence_permutation'],memory['nulls']['tau']));m2,md=token_logits(q1,q4,r1[:1],r4[:1]);out['M2']=m2;diag['M2_alpha']=md['alpha']
  for k,z in out.items():parts[k].append(target_lp(z,pair[1][begin:end]))
  for k,v in diag.items():packed.setdefault(k,[]).extend(v.cpu().tolist())
 sums={k:float(torch.cat(v).sum()) for k,v in parts.items()};late=expected['original_methods']['native_k4']['normalized_scores'][index]-expected['original_methods']['native_k5']['normalized_scores'][index];scores={k:v/int(chars)+late for k,v in sums.items()}
 for name,old in [('M2','M2_N0_token'),('M3','M3_multi_token')]:assert scores[name]==expected['method_scores'][old][index] and packed[name+'_alpha']==expected['token_diagnostics'][index][old]['alpha']
 assert scores['frequency']==scores['M3']
 return dict(choice=index,characters=int(chars),tokens=len(pair[1]),sums=sums,scores=scores,late=late,metrics=packed,retrieval=retrieval,readout_parity=parity,original_sums=original,reference_sums=reference_sums,baseline_parity_error=0,reference_positions=maps,request_sha256=digest(request),target_ids_sha256=digest(pair[1]))

def main():
 started=time.monotonic();spec=gate();memory=read('configs/r25_MEMORY_MANIFEST.json')
 from r19_common import inputs
 data=inputs(spec);authority={r['id']:r for split in ['discovery','selection'] for r in jsonlines('results/r7_'+split+'_development.jsonl')}
 from runtime import load_model,check_budget
 from ouro_cached_scoring import OuroCachedScorer
 from weight_edits import tensor_sha256
 from r25_keys import audit_inputs_before_model,build
 import torch
 from transformers import AutoTokenizer
 audit_tok=AutoTokenizer.from_pretrained(str(ROOT/'models/Ouro-1.4B'),local_files_only=True,use_fast=True);assert audit_tok.is_fast;tokens,input_audit=audit_inputs_before_model(audit_tok,memory,data);save('results/r25_INPUT_AUDIT.json',input_audit);del audit_tok
 assert not (ROOT/'results/r25_pilot.jsonl').exists();t=time.monotonic();tok,model=load_model();scorer=OuroCachedScorer(model,5);load_seconds=time.monotonic()-t;identity=spec['model_identity'];assert sha('models/Ouro-1.4B/model.safetensors')==identity['safetensors_sha256']
 for p,h in identity['metadata_and_tokenizer_sha256'].items():assert sha('models/Ouro-1.4B/'+p)==h
 before={n:tensor_sha256(p) for n,p in model.named_parameters()};torch.cuda.reset_peak_memory_stats();assert model.config.vocab_size==memory['vocabulary_size'];timing={k:0. for k in ['key_build','key_audit','Q_head','N_forward','N_head','retrieval','checks']};keys,key_logits,keyaudit=build(model,scorer,tokens,memory,timing);count=0;peak=torch.cuda.max_memory_allocated()
 save('results/r25_KEY_AUDIT.json',keyaudit)
 with (ROOT/'results/r25_pilot.jsonl').open('x',encoding='utf8') as out:
  for chosen,clean,expected,state in data:
   check_budget(5);assert time.monotonic()<float(os.environ['R25_SOFT_MONOTONIC']);payload=torch.load(ROOT/state['file'],map_location='cpu',weights_only=True);assert payload['id']==chosen['id'];expected['method_scores']=authority[chosen['id']]['method_scores']
   candidates=[measure(model,scorer,tok,req,payload['candidate_states'][j],expected,j,clean['choice_char_lengths'][j],keys,key_logits,memory,tokens,timing,count==0 and j==0) for j,req in enumerate(clean['request_pairs'])]
   t=time.monotonic();from r7_inference import methods_from_sums
   from r7_core import baseline_vectors
   lengths=np.asarray(clean['choice_char_lengths']);orig=methods_from_sums({k:[c['original_sums'][k] for c in candidates] for k in candidates[0]['original_sums']},lengths);prior=methods_from_sums({k:[c['reference_sums'][0][k] for c in candidates] for k in candidates[0]['reference_sums'][0]},lengths);baselines=baseline_vectors(orig,prior);assert all(v.tolist()==authority[chosen['id']]['baseline_scores'][k] for k,v in baselines.items());timing['checks']+=time.monotonic()-t
   out.write(json.dumps(dict(id=chosen['id'],split=chosen['split'],original_baseline_vectors_parity_error=0,candidates=candidates),allow_nan=False)+'\n');out.flush();os.fsync(out.fileno());count+=1;peak=max(peak,torch.cuda.max_memory_allocated());save('results/r25_PROGRESS.json',dict(completed=count,target=24,elapsed_seconds=time.monotonic()-started,component_seconds=timing))
 after={n:tensor_sha256(p) for n,p in model.named_parameters()};assert before==after and count==24
 save('results/r25_COMPLETE.json',dict(status='complete',n=count,protocol_sha256=sha(LOCK),score_file_sha256=sha('results/r25_pilot.jsonl'),all_parameter_hashes_before=before,all_parameter_hashes_after=after,weights_modified=False,elapsed_seconds=time.monotonic()-started,model_load_seconds=load_seconds,component_seconds=timing,key_audit_sha256=sha('results/r25_KEY_AUDIT.json'),input_audit_sha256=sha('results/r25_INPUT_AUDIT.json'),peak_cuda_allocated_bytes=peak,memory_scope='GB10 unified allocator only',timing_scope='cachedmain exploratorypilot, not deployment throughput',heldout_accessed=False,KV_cache=False,explicit_attention_mask=False))
if __name__=='__main__':main()
