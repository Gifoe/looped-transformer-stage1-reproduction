"""Tiny synthetic CPU BF16 sanity test; never a substitute for GPU parity."""
import sys,torch
from pathlib import Path
from transformers import AutoModelForCausalLM
from native_capture import capture,candidate_logprob
P=Path(__file__).resolve();R=P.parents[2];ROOT=R.parent
sys.path.insert(0,str(R/'experiments/ouro_weight_space_repair_v2/scripts'))
from ouro_cached_scoring import OuroCachedScorer
from r7_inference import target_lp
torch.set_num_threads(4)
m=AutoModelForCausalLM.from_pretrained(ROOT/'model',trust_remote_code=True,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').eval();s=OuroCachedScorer(m,4);x=torch.tensor([[330,389]]);target=[340]
with torch.inference_mode():
 _,historical,_=s._run(x[0].tolist(),positions=[1],use_cache=False)
 plain=m(x,labels=None,use_cache=False,exit_at_step=3).logits
 with capture(m,[0,1],4) as states: read=m(x,labels=None,use_cache=False,exit_at_step=3).logits
 assert torch.equal(plain,read)
 with capture(m,[0,1],4,identity=True) as states2: identity=m(x,labels=None,use_cache=False,exit_at_step=3).logits
 assert torch.equal(plain,identity)
 for k in range(4):
  old=float(target_lp(m.lm_head(historical[k])[0].float(),target).sum());new=candidate_logprob(m,states['loop_post_norm'][k],target,[1]);assert old==new,(k,old,new)
  if k<3:
   m.model.total_ut_steps=k+1
   try: independent=m(x,labels=None,use_cache=False,exit_at_step=k).logits
   finally: m.model.total_ut_steps=4
   assert torch.equal(independent,m.lm_head(states['loop_post_norm'][k].unsqueeze(0)))
print('PASS tiny CPU only: read-only, identity, historical scorer, independent K1-K3')
