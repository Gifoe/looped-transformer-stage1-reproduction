import torch,hashlib
@torch.inference_mode()
def match(before,delta,targets,seed):
 # rowsxflattenedheads; targets are actual T message writeback norms, no gold.
 n,d=before.shape;g=torch.Generator().manual_seed(seed);perm=torch.randperm(d,generator=g).cuda();direction=delta.double()[:,perm];zero=direction.norm(dim=-1)==0
 if bool(zero.any()):
  fallback=torch.randn((n,d),generator=g,dtype=torch.float64).cuda();direction=torch.where(zero.unsqueeze(-1),fallback,direction)
 direction/=direction.norm(dim=-1,keepdim=True).clamp_min(1e-30);b=before.double();target=torch.as_tensor(targets,device=before.device,dtype=torch.float64);lo=torch.zeros_like(target);hi=torch.maximum(2*target,torch.full_like(target,1e-12));best=before.clone();error=target.clone();real=torch.zeros_like(target);amplitude=torch.zeros_like(target)
 for step in range(40):
  a=hi if step<8 else(lo+hi)/2;z=(b+a.unsqueeze(-1)*direction).to(before.dtype);norm=(z.double()-b).norm(dim=-1);err=(norm-target).abs();take=err<error;best=torch.where(take.unsqueeze(-1),z,best);real=torch.where(take,norm,real);error=torch.where(take,err,error);amplitude=torch.where(take,a,amplitude)
  if step<8:hi=torch.where(norm<target,hi*2,hi)
  else:lo=torch.where(norm<target,a,lo);hi=torch.where(norm>=target,a,hi)
 return best,{'target':target.cpu().tolist(),'actual':real.cpu().tolist(),'relative_error':(error/target.clamp_min(1e-30)).cpu().tolist(),'matched':(error<=torch.maximum(target*.01,torch.full_like(target,1e-9))).cpu().tolist(),'amplitude':amplitude.cpu().tolist(),'zero_direction_fallback_rows':int(zero.sum())}
