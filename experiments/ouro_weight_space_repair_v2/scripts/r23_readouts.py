"""R23 exact native2/3 full-vocabulary head readouts, original scorer unchanged."""
def heads(model,states,keys):
 import torch
 result={}
 for k in keys:
  h=states[k-1].to(model.lm_head.weight.device);assert h.dtype==torch.bfloat16 and h.ndim==3 and h.shape[0]==1
  z=model.lm_head(h)[0];assert z.dtype==torch.bfloat16;result[k]=z.float()
 return result
