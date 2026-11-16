import torch
@torch.inference_mode()
def apply(before,targets,seed):
 n,d=before.shape;g=torch.Generator().manual_seed(seed);direction=torch.randn((n,d),generator=g,dtype=torch.float64).to(before.device);direction/=direction.norm(dim=-1,keepdim=True);base=before.double();target=torch.as_tensor(targets,device=before.device,dtype=torch.float64);lo=torch.zeros_like(target);hi=torch.maximum(2*target,torch.full_like(target,1e-12));best=before.clone();err=target.clone();real=torch.zeros_like(target)
 for step in range(40):
  a=hi if step<8 else (lo+hi)/2;proposal=(base+a.unsqueeze(-1)*direction).to(before.dtype);norm=(proposal.double()-base).norm(dim=-1);error=(norm-target).abs();take=error<err;best=torch.where(take.unsqueeze(-1),proposal,best);err=torch.where(take,error,err);real=torch.where(take,norm,real)
  if step<8:hi=torch.where(norm<target,hi*2,hi)
  else:lo=torch.where(norm<target,a,lo);hi=torch.where(norm>=target,a,hi)
 return best,{'seed':seed,'target':target.cpu().tolist(),'actual':real.cpu().tolist(),'relative_error':(err/target.clamp_min(1e-30)).cpu().tolist(),'matched':(err<=torch.maximum(target*.01,torch.full_like(target,1e-9))).cpu().tolist()}
