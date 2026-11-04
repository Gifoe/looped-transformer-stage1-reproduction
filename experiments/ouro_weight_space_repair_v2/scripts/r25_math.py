"""R25 fixed native-fullV Hellinger occurrence1NN and value nulls; target-free."""
import numpy as np

def null_maps(values,vocab):
 values=np.asarray(values,dtype=np.int64);assert values.ndim==1 and len(values)>0 and values.min()>=0 and values.max()<vocab
 raw=np.random.default_rng(2021).permutation(len(values));counts=np.bincount(values,minlength=vocab);tau=np.arange(vocab);rng=np.random.default_rng(2021);layers=[]
 for n in sorted(set(counts[counts>0].tolist())):
  ids=np.flatnonzero(counts==n);image=rng.permutation(ids);tau[ids]=image;layers.append(dict(frequency=int(n),size=len(ids),fixed_points=int((ids==image).sum())))
 assert np.array_equal(np.sort(tau),np.arange(vocab)) and np.array_equal(np.bincount(tau[values],minlength=vocab),counts)
 return dict(raw_occurrence_permutation=raw,tau=tau,layers=layers,raw_fixed_occurrences=int((raw==np.arange(len(values))).sum()),raw_unchanged_values=int((values[raw]==values).sum()),tau_unchanged_occurrence_values=int((tau[values]==values).sum()),value_counts=counts)

def numpy_retrieve(key_logits,query_logits):
 k=np.asarray(key_logits,dtype=np.float64);q=np.asarray(query_logits,dtype=np.float64)
 if k.ndim!=2 or q.ndim!=2 or k.shape[0]==0 or k.shape[1]!=q.shape[1] or k.shape[1]<2 or not np.isfinite(k).all() or not np.isfinite(q).all():raise ValueError('Invalid key/query logits')
 def sq(z):
  p=np.exp(z-z.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True);return np.sqrt(p)
 keys=sq(k);queries=sq(q);dist=np.stack([np.sum((keys-a)**2,axis=-1) for a in queries]);return np.argmin(dist,axis=-1),dist

def torch_sqrt(z):
 import torch
 if z.ndim!=2 or z.shape[-1]<2 or not bool(torch.isfinite(z).all()):raise ValueError('Invalid fullV head logits')
 p=torch.softmax(z.double(),dim=-1)
 if not bool(torch.isfinite(p).all()) or bool((p<0).any()) or bool((p.sum(-1)-1).abs().gt(1e-12).any()):raise ValueError('Invalid fullV probability')
 return p.sqrt()

def torch_retrieve(keys,query_logits):
 import torch
 if keys.ndim!=2 or not len(keys) or keys.shape[1]!=query_logits.shape[-1] or keys.dtype!=torch.float64 or not bool(torch.isfinite(keys).all()) or bool((keys<0).any()) or bool((keys.square().sum(-1)-1).abs().gt(1e-12).any()):raise ValueError('Invalid sqrt keys')
 queries=torch_sqrt(query_logits);all_dist=[]
 for query in queries:
  chunks=[]
  for begin in range(0,len(keys),8):chunks.append((keys[begin:begin+8]-query).square().sum(-1))
  all_dist.append(torch.cat(chunks))
 distances=torch.stack(all_dist);neighbors=distances.argmin(-1);two=distances.sort(-1).values[:,:2];gap=two[:,1]-two[:,0] if len(keys)>1 else torch.full((len(queries),),float('inf'),device=keys.device)
 return neighbors,distances,gap

def norm_matched(c,direction,p):
 import torch
 def center(x):return x-(p*x).sum(-1,keepdim=True)
 a=center(c);b=center(direction);A=(p*a.square()).sum(-1);B=(p*b.square()).sum(-1);t=64*torch.finfo(torch.float64).eps*b.abs().amax(-1).square();bad=(A>0)&(B<=t);ratio=torch.where(B>t,A/torch.where(B>t,B,torch.ones_like(B)),torch.zeros_like(B));update=b*ratio.sqrt().unsqueeze(-1)
 return update,bad,A,(p*update.square()).sum(-1)

def torch_logits(q1,q4,refs1,refs4,neighbors,values,raw_permutation,tau):
 import torch
 from r7_math import token_logits
 base,bd=token_logits(q1,q4,refs1,refs4);v=q4.shape[-1];device=q4.device;values=torch.as_tensor(values,device=device,dtype=torch.long);raw=torch.as_tensor(raw_permutation,device=device,dtype=torch.long);tau=torch.as_tensor(tau,device=device,dtype=torch.long)
 def mean(x):return x[0] if torch.equal(x,x[:1].expand_as(x)) else x.mean(0)
 def onehot(ids):return torch.nn.functional.one_hot(ids,num_classes=v).double()
 r=onehot(values[neighbors]);c=mean(r[0].unsqueeze(0)-r[1:]);rs=onehot(values[raw][neighbors]);cs=mean(rs[0].unsqueeze(0)-rs[1:]);rt=onehot(tau[values][neighbors]);ct=mean(rt[0].unsqueeze(0)-rt[1:]);permuted=torch.zeros_like(c);permuted[...,tau]=c;assert torch.equal(ct,permuted) and torch.equal((ct!=0).sum(-1),(c!=0).sum(-1))
 p=torch.softmax(q4.double(),dim=-1);matched,bad,A,B=norm_matched(c,ct,p)
 def output(x):
  if bool((x==0).all()):return base
  return torch.where((x==0).all(-1,keepdim=True),base,(base.double()+x).float())
 f=torch.bincount(values,minlength=v).double()/len(values);fc=mean(torch.stack([f-f]*3));assert bool((fc==0).all())
 result=dict(primary=output(c),M3=base,q_only=output(r[0]),reference_only=output(-mean(r[1:])),memory_direction_only=c.float(),frequency=base,raw_occurrence_shuffle=output(cs),freq_bijection_matched=output(matched))
 for j in range(3):result['leaveout_'+str(j)]=output(mean(r[0].unsqueeze(0)-r[[i+1 for i in range(3) if i!=j]]))
 diag=dict(M3_alpha=bd['alpha'],c_zero=(c==0).all(-1),raw_c_zero=(cs==0).all(-1),tau_c_zero=(ct==0).all(-1),c_nonzero_coordinates=(c!=0).sum(-1),main_P_norm2=A,matched_P_norm2=B,unmatched=bad,query_reference_same_value=(values[neighbors[1:]]==values[neighbors[0]]).sum(0))
 assert all(bool(torch.isfinite(z).all()) for z in result.values());return result,diag
