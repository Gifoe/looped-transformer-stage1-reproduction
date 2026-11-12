"""Deterministic byte hash of live model parameters and buffers."""
import hashlib,torch
def model_digest(m):
 h=hashlib.sha256()
 for n,t in list(m.named_parameters())+list(m.named_buffers()):
  h.update(n.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
 return h.hexdigest()
