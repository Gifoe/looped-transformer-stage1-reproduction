"""R23 SLED-inspired analytic full-V one-step targets; no targets/gold/model."""
import numpy as np

def numpy_evolution(z):
 z=np.asarray(z,dtype=np.float64)
 if z.shape[0]!=4 or z.shape[-1]<2 or not np.isfinite(z).all():raise ValueError('Expected four finite fullV logits')
 p=np.exp(z-z.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True);d=z[:3]-z[3];d-=d.mean(-1,keepdims=True);d2=(d*d).sum(-1,keepdims=True);dt=64*np.finfo(np.float64).eps*np.max(np.abs(d),axis=-1,keepdims=True)**2
 pk=p[:3];gn=(1-pk)**2+np.maximum((pk*pk).sum(-1,keepdims=True)-pk*pk,0);gt=64*np.finfo(np.float64).eps*np.maximum(1.,np.max(pk*pk,axis=-1,keepdims=True));bad=(d2<=dt)|(gn<=gt)
 numerator=(d*pk).sum(-1,keepdims=True)-d;den=np.sqrt(d2*gn);a=np.divide(numerator,np.where(bad,1.,den));b=np.where(bad,0.,np.maximum(a,0.)**2);weight=b.sum(0);total=weight.sum(-1,keepdims=True);latent=np.divide(weight,np.where(total==0,1.,total));latent=np.where(total==0,p[3],latent);u=np.where(total==0,0.,latent-p[3])
 if not all(np.isfinite(x).all() for x in [latent,u,b]):raise ValueError('Nonfinite evolution')
 return u,dict(latent=latent,p4=p[3],effective_support=1/(latent*latent).sum(-1),TV=.5*np.abs(u).sum(-1),zero=(total[...,0]==0),degenerate_layers=(d2[...,0]<=dt[...,0]).sum(0),degenerate_gradients=bad.sum(axis=(0,-1)))

def torch_evolution(z):
 import torch
 with torch.no_grad():
  z=z.double()
  if z.shape[0]!=4 or z.shape[-1]<2 or not bool(torch.isfinite(z).all()):raise ValueError('Expected four finite fullV logits')
  p=torch.softmax(z,dim=-1);d=z[:3]-z[3];d=d-d.mean(-1,keepdim=True);d2=d.square().sum(-1,keepdim=True);dt=64*torch.finfo(torch.float64).eps*d.abs().amax(-1,keepdim=True).square();pk=p[:3]
  gn=(1-pk).square()+(pk.square().sum(-1,keepdim=True)-pk.square()).clamp_min(0);gt=64*torch.finfo(torch.float64).eps*torch.maximum(torch.ones_like(d2),pk.square().amax(-1,keepdim=True));bad=(d2<=dt)|(gn<=gt)
  numerator=(d*pk).sum(-1,keepdim=True)-d;den=(d2*gn).sqrt();a=numerator/torch.where(bad,torch.ones_like(den),den);b=torch.where(bad,torch.zeros_like(a),a.clamp_min(0).square());weight=b.sum(0);total=weight.sum(-1,keepdim=True);latent=weight/torch.where(total==0,torch.ones_like(total),total);latent=torch.where(total==0,p[3],latent);u=torch.where(total==0,torch.zeros_like(latent),latent-p[3])
  if not all(bool(torch.isfinite(x).all()) for x in [latent,u,b]):raise ValueError('Nonfinite evolution')
  return u,dict(latent=latent,p4=p[3],effective_support=1/latent.square().sum(-1),TV=.5*u.abs().sum(-1),zero=(total[...,0]==0),degenerate_layers=(d2[...,0]<=dt[...,0]).sum(0),degenerate_gradients=bad.sum(dim=(0,-1)))

def torch_logits(q,refs,permutation):
 import torch
 from r7_math import token_logits
 base,base_diag=token_logits(q[0],q[3],refs[:,0],refs[:,3]);uq,dq=torch_evolution(q);items=[torch_evolution(r) for r in refs];ur=torch.stack([x[0] for x in items]);p=dq['p4']
 def mean(x):return x[0] if torch.equal(x,x[:1].expand_as(x)) else x.mean(0)
 nr=mean(ur);c=mean(uq.unsqueeze(0)-ur)
 def center(x):return x-(p*x).sum(-1,keepdim=True)
 cc=center(c);nc=(p*cc.square()).sum(-1)
 def matched(direction):
  dc=center(direction);norm=(p*dc.square()).sum(-1);threshold=64*torch.finfo(torch.float64).eps*dc.abs().amax(-1).square();bad=(nc>0)&(norm<=threshold);ratio=torch.where(norm>threshold,nc/torch.where(norm>threshold,norm,torch.ones_like(norm)),torch.zeros_like(norm));update=dc*ratio.sqrt().unsqueeze(-1);return update,bad,(p*update.square()).sum(-1)
 shuffle,sd,sn=matched(cc[...,torch.as_tensor(permutation,device=c.device)]);prob,ed,en=matched(mean(torch.stack([x[1]['p4']-p for x in items])))
 def output(x):return torch.where((x==0).all(-1,keepdim=True),base,(base.double()+x).float())
 values=dict(primary=output(c),M3=base,q_only=output(uq),reference_only=output(-nr),shuffle=output(shuffle),probability_contrast=output(prob))
 for j in range(3):values['leaveout_'+str(j)]=output(mean(uq.unsqueeze(0)-ur[[i for i in range(3) if i!=j]]))
 diag=dict(M3_alpha=base_diag['alpha'],c_P_norm2=nc,shuffle_P_norm2=sn,probability_P_norm2=en,shuffle_unmatched=sd,probability_unmatched=ed,c_zero=(c==0).all(-1),Q_support=dq['effective_support'],Q_TV=dq['TV'],Q_u_P_norm2=(p*uq.square()).sum(-1))
 for j,(uj,d) in enumerate(items):
  diag['N'+str(j)+'_u_P_norm2']=(d['p4']*uj.square()).sum(-1)
  for key in ['effective_support','TV','zero','degenerate_layers','degenerate_gradients']:diag['N'+str(j)+'_'+key]=d[key]
 for key in ['zero','degenerate_layers','degenerate_gradients']:diag['Q_'+key]=dq[key]
 assert all(bool(torch.isfinite(v).all()) for v in values.values())
 return values,diag,(uq,ur,c,dq['latent'])
