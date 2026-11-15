"""Apply original E2 frozen heads at their actual authorized R sites only.

Requires separately attested E2_FROZEN_DECODER from the existing research project.
Does not fit, select, read gold, call a model or claim native causal use.
"""
import numpy as np,torch
from E2_FROZEN_DECODER import FrozenE2Decoders,MODEL_KEYS
PRIMARY={'l18_input_raw_trainK1_real','l18_input_unit_l2_trainK1_real','l18_input_raw_trainK1_permuted','l18_input_unit_l2_trainK1_permuted','post_norm_raw_trainK1_real','post_norm_unit_l2_trainK1_real','post_norm_raw_trainK1_permuted','post_norm_unit_l2_trainK1_permuted'}
def apply_frozen_heads(path,stored,targets,condition_names,*,all_models=False):
 heads=FrozenE2Decoders(path);out={};keys=MODEL_KEYS if all_models else [k for k in MODEL_KEYS if k in PRIMARY]
 sites={'K1':{'l18_input':stored['source_K1_L18_x_R'],'post_norm':stored['source_B'][:,-1,:]}}
 for name in condition_names:
  if name+'_L18_x_R' in stored and name+'_finalnorm_R' in stored:sites[name]={'l18_input':stored[name+'_L18_x_R'],'post_norm':stored[name+'_finalnorm_R']}
 for name,by_site in sites.items():
  out[name]={}
  for k in keys:
   t=by_site['l18_input' if k.startswith('l18_input_') else 'post_norm'];assert t.dtype==torch.bfloat16 and tuple(t.shape)==(1,2048) and torch.isfinite(t).all();x=t.float().cpu().numpy().astype(np.float64);v=heads.predict(x,k,targets);out[name][k]={'parameter_sha256':v['parameter_sha256'],'offered_token_ids':v['offered_token_ids'],'scores':v['scores'].tolist(),'probabilities':v['probabilities'].tolist(),'prediction_offered_index':v['prediction_offered_index'].tolist(),'top_ties_offered_indices':v['top_ties_offered_indices']}
 return {'status':'FROZEN_E2_DESCRIPTIVE_APPLICATION','primary_keys':sorted(PRIMARY),'secondary_all16':all_models,'scope':'preexisting supervised frozen heads; no fitting and no native causal-use inference','results':out}
