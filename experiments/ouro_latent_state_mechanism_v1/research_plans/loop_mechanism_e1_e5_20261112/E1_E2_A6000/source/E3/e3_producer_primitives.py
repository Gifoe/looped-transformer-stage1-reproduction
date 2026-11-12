"""CPU-only primitives for producer Q-token splice and post-RoPE K_Q exchange."""
import torch

def _valid_pair(a,b):
 return torch.is_tensor(a) and torch.is_tensor(b) and a.shape==b.shape and a.dtype==b.dtype and a.device==b.device and torch.isfinite(a).all() and torch.isfinite(b).all()
def splice_hidden_Q(receiver,donor,R,prefix_length):
 if not _valid_pair(receiver,donor) or receiver.ndim!=3 or receiver.shape[0]!=1:raise ValueError('hidden source mismatch')
 L=receiver.shape[1]
 if type(R)!=int or not 0<prefix_length<R==L-1:raise ValueError('boundary')
 out=receiver.clone();out[:,prefix_length:R,:]=donor[:,prefix_length:R,:]
 if not torch.equal(out[:,:prefix_length,:],receiver[:,:prefix_length,:]) or not torch.equal(out[:,R:,:],receiver[:,R:,:]) or not torch.equal(out[:,prefix_length:R,:],donor[:,prefix_length:R,:]):raise ValueError('hidden outside Q changed')
 return out
def splice_postrope_K_Q(receiver,donor,R,prefix_length):
 if not _valid_pair(receiver,donor) or receiver.ndim!=4 or receiver.shape[0]!=1:raise ValueError('key source mismatch')
 L=receiver.shape[-2]
 if type(R)!=int or not 0<prefix_length<R==L-1:raise ValueError('boundary')
 out=receiver.clone();out[...,prefix_length:R,:]=donor[...,prefix_length:R,:]
 if not torch.equal(out[...,:prefix_length,:],receiver[...,:prefix_length,:]) or not torch.equal(out[...,R:,:],receiver[...,R:,:]) or not torch.equal(out[...,prefix_length:R,:],donor[...,prefix_length:R,:]):raise ValueError('K outside Q changed')
 return out
