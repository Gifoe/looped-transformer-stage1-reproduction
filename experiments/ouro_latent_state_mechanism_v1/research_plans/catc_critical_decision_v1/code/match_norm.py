"""Outcome-free scalar search for an actual BF16 correction norm, never exceeding cap."""
import torch
@torch.inference_mode()
def match_norm(before,direction,target,cap):
 d=direction.double();d=d/d.norm().clamp_min(1e-30);target=float(target);cap=float(cap);aim=min(target,cap)
 def probe(a):
  z=(before.double()+a*d).to(before.dtype);return z,float((z.double()-before.double()).norm())
 lo=0.;hi=max(aim*1.25,1e-12);candidates=[(before.clone(),0.,0.)]
 for _ in range(32):
  z,n=probe(hi)
  if n<=cap:candidates.append((z,n,hi))
  if n>=aim:break
  hi*=2
 for _ in range(32):
  mid=(lo+hi)/2;z,n=probe(mid)
  if n<=cap:candidates.append((z,n,mid))
  if n<aim:lo=mid
  else:hi=mid
 z,n,amplitude=min(candidates,key=lambda x:(abs(x[1]-target),x[2]));error=abs(n-target);matched=error<=max(1e-9,.01*target)
 assert n<=cap
 return z,{'target_actual_norm':target,'actual_norm':n,'absolute_error':error,'relative_error':error/max(target,1e-30),'cap':cap,'matched_within_1percent':matched,'amplitude':amplitude,'target_exceeds_current_cap':target>cap}
