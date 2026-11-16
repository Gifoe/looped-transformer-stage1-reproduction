"""Deployment readout: fixed coefficients, no gold or learned controller. Zero is native."""
import contextlib,hashlib,json
import CATC_ADAPTATION as C
from CATC_ADAPTATION import torch
@torch.inference_mode()
def predict_static(model,q,alpha,kind='structured'):
 assert len(alpha)==14 and not model.training and model.model.total_ut_steps==4
 assert hashlib.sha256(json.dumps(q['ids']).encode()).hexdigest()==q['input_ids_sha256']
 ids=torch.tensor([q['ids']],device='cuda')
 context=contextlib.nullcontext() if not any(alpha) else C.control(model,q,kind=kind,coefficients=alpha)
 with context:_,states,_=model.model(ids,use_cache=False,past_key_values=None)
 return model.lm_head(states[3])[:,q['R']]
