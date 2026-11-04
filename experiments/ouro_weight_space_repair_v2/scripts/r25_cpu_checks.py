"""R25 synthetic-only fullV retrieval/null/zero/matching invariants."""
import numpy as np
from r25_math import null_maps,numpy_retrieve,torch_sqrt,torch_retrieve,torch_logits,norm_matched

def main():
 import torch,inspect
 torch.set_num_threads(2);rng=np.random.default_rng(2021);k=rng.normal(size=(11,9)).astype(np.float32);k[1]=k[0];q=np.concatenate([k[:1],rng.normal(size=(2,9)).astype(np.float32)]);keys=torch_sqrt(torch.from_numpy(k));ix,dist,gap=torch_retrieve(keys,torch.from_numpy(q));ni,nd=numpy_retrieve(k,q);assert np.array_equal(ix.numpy(),ni) and np.max(np.abs(dist.numpy()-nd))<=1e-12 and int(ix[0])==0
 values=np.array([0,0,1,1,2,3,4,5,6,7,8]);maps=null_maps(values,9);assert np.array_equal(np.sort(maps['tau']),np.arange(9)) and np.array_equal(np.bincount(maps['tau'][values],minlength=9),np.bincount(values,minlength=9))
 # Singleton-frequency class and identity mapping are legitimate, never resampled.
 singleton=null_maps(np.array([0,0,1,2,2,2]),4);assert singleton['tau'][0]==0 and singleton['tau'][1]==1 and singleton['tau'][2]==2 and singleton['tau'][3]==3
 q1=torch.from_numpy(rng.normal(size=(3,9)).astype(np.float32));q4=torch.from_numpy(q);r1=q1.unsqueeze(0).repeat(3,1,1);r4=q4.unsqueeze(0).repeat(3,1,1);neighbors=torch.tensor([[0,2,3],[2,3,4],[3,4,5],[4,5,6]])
 out,diag=torch_logits(q1,q4,r1,r4,neighbors,values,maps['raw_occurrence_permutation'],maps['tau']);assert torch.max(torch.abs(diag['main_P_norm2']-diag['matched_P_norm2']))<=1e-10 and out['frequency'] is out['M3']
 z=torch_logits(q1,q4,r1,r4,torch.zeros_like(neighbors),values,maps['raw_occurrence_permutation'],maps['tau']);assert z[0]['primary'] is z[0]['M3'] and bool(z[1]['c_zero'].all())
 identity=torch_logits(q1,q4,r1,r4,neighbors,values,maps['raw_occurrence_permutation'],np.arange(9));assert torch.max(torch.abs(identity[1]['main_P_norm2']-identity[1]['matched_P_norm2']))<=1e-10 and torch.equal(identity[1]['c_zero'],identity[1]['tau_c_zero'])
 # Exact coordinate permutation, zero qualification and value multiset.
 for _ in range(20):
  nn=torch.from_numpy(rng.integers(0,len(values),size=(4,3)));v,d=torch_logits(q1,q4,r1,r4,nn,values,maps['raw_occurrence_permutation'],maps['tau']);assert torch.equal(d['c_zero'],d['tau_c_zero'])
 # Raw occurrence shuffle can collapse nonzero c, unlike ID bijection.
 vv=np.array([0,1,0,2]);nn=torch.tensor([[0],[1],[1],[1]]);base=q1[:1];m= torch_logits(base,q4[:1],r1[:,:1],r4[:,:1],nn,vv,np.array([0,2,1,3]),np.arange(9));assert not bool(m[1]['c_zero'][0]) and bool(m[1]['raw_c_zero'][0])
 c=torch.tensor([[1.,-1.,0.,0.]],dtype=torch.float64);direction=torch.tensor([[0.,0.,1.,-1.]],dtype=torch.float64);p=torch.tensor([[.5,.5,0.,0.]],dtype=torch.float64);update,bad,A,B=norm_matched(c,direction,p);assert bool(bad[0]) and bool((update==0).all())
 ext=torch_logits(torch.cat([q1,q1[:1]]),torch.cat([q4,q4[:1]]),torch.cat([r1,r1[:,:1]],1),torch.cat([r4,r4[:,:1]],1),torch.cat([neighbors,neighbors[:,:1]],1),values,maps['raw_occurrence_permutation'],maps['tau']);assert torch.equal(out['primary'],ext[0]['primary'][:3])
 reordered=torch_logits(q1,q4,r1[[2,0,1]],r4[[2,0,1]],neighbors[[0,3,1,2]],values,maps['raw_occurrence_permutation'],maps['tau']);assert torch.max(torch.abs(out['primary']-reordered[0]['primary']))<=2e-5
 badlogits=k.copy();badlogits[0,0]=np.nan
 try:numpy_retrieve(badlogits,q)
 except ValueError:pass
 else:raise AssertionError('Nonfinite accepted')
 assert all('target' not in inspect.signature(f).parameters for f in [torch_retrieve,torch_logits,numpy_retrieve])
 from r25_keys import audit_inputs_before_model
 from r25_common import digest
 import runtime
 class FakeTokenizer:
  def encode(self,text,add_special_tokens=False):return list(range(146))
  def __call__(self,text,**kwargs):return {'input_ids':list(range(146)),'offset_mapping':[(i,i+1) for i in range(146)]}
 memory={'key_text':'prefixQuestion:','entries':[{'source_token_position':i,'key_prediction_position':i-1,'value_id':i,'offset':[i,i+1]} for i in range(1,144)]}
 synthetic=[({'id':str(i)},{'request_pairs':[('prefixQuestion: synthetic',' continuation')]},{},{'target_id_sha256':[digest([999])],'choice_token_counts':[1]}) for i in range(24)]
 original_encoder=runtime.encode_pair
 try:
  runtime.encode_pair=lambda tok,context,continuation:(list(range(146)),[999])
  toks,audit=audit_inputs_before_model(FakeTokenizer(),memory,synthetic);assert audit['n']==24 and audit['candidates']==24 and not audit['model_or_key_or_query_loaded']
  broken=list(synthetic);broken[0]=({'id':'0'},{'request_pairs':[('different prefix',' continuation')]},{},{'target_id_sha256':[digest([999])],'choice_token_counts':[1]})
  try:audit_inputs_before_model(FakeTokenizer(),memory,broken)
  except AssertionError:pass
  else:raise AssertionError('Broken shared prefix accepted')
 finally:runtime.encode_pair=original_encoder
 from pathlib import Path
 source=Path(__file__).with_name('r25_worker.py').read_text(encoding='utf-8-sig');assert source.index('tokens,input_audit=audit_inputs_before_model')<source.index('tok,model=load_model')
 analysis=Path(__file__).with_name('r25_analyze.py').read_text(encoding='utf-8-sig');assert analysis.index('audit=integrity(s,d,u,ka,rows)')<analysis.index("old={r['id']:r") and analysis.index('normerror<=1e-10')<analysis.index('def main():')
 return dict(synthetic_all24_pre_model_token_audit_and_bad_prefix_rejection=True,pre_model_CPU_audit_source_order=True,complete_label_free_before_gold_source_order=True,status='passed_CPU_synthetic',distance_maxerror=float(np.max(np.abs(dist.numpy()-nd))),direct_fullV=True,exact_tie_earliest=True,duplicate_keys_values=True,value_histograms_preserved=True,tau_bijective=True,singleton_fixed_identity_accepted=True,tau_coordinate_and_zero_qualification_exact=True,raw_collision_counterexample=True,frequency_exact_M3=True,c_zero_exact_M3_object=True,reference_permutation=True,candidate_extension_exact=True,normmatch=True,extreme_probability_unmatched=True,nonfinite_rejected=True,target_free_signatures=True,scope='Synthetic only, no real keys/query/model/GPU')
if __name__=='__main__':
 import json
 print(json.dumps(main()))
