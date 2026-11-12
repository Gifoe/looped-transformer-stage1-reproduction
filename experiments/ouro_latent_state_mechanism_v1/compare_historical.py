"""Real validation samples, historical scorer vs capture; not a full parity gate."""
import argparse,json,sys
from pathlib import Path
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from prompt_spans import make_question,token_spans
from native_capture import capture,candidate_logprob
P=Path(__file__).resolve();R=P.parents[2];ROOT=R.parent
sys.path.insert(0,str(R/'experiments/ouro_weight_space_repair_v2/scripts'))
from r10_core import joint_prefix,question_prompt
from ouro_cached_scoring import OuroCachedScorer
from r7_inference import target_lp

def main():
 p=argparse.ArgumentParser();p.add_argument('--count',type=int,default=3);p.add_argument('--device',choices=['cuda','cpu'],default='cuda');a=p.parse_args()
 if not 1<=a.count<=299: raise ValueError('invalid count')
 if a.device=='cuda' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):raise RuntimeError('CUDA BF16 required')
 tok=AutoTokenizer.from_pretrained(ROOT/'model',local_files_only=True,trust_remote_code=True)
 m=AutoModelForCausalLM.from_pretrained(ROOT/'model',local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to(a.device).eval()
 m.config.use_cache=False;m.model.config.use_cache=False
 scorer=OuroCachedScorer(m,4);demo=json.loads((R/'experiments/ouro_weight_space_repair_v2/data/slcd_fixed_demonstrations.json').read_text());prefix=joint_prefix(demo)
 data=pq.read_table(ROOT/'data/validation.parquet').to_pylist()[:a.count];results=[]
 with torch.inference_mode():
  for row in data:
   prompt,spans=make_question(prefix,row['question'],row['choices']['text'],question_prompt)
   for j in range(len(row['choices']['text'])):
    enc=token_spans(tok,prompt,' '+chr(65+j),spans);positions=enc['predecessor_positions'];ids=enc['input_ids'];target=enc['target_ids']
    _,historical,_=scorer._run(ids,positions=positions,use_cache=False)
    with capture(m,positions,4) as current:m(torch.tensor([ids],device=a.device),labels=None,use_cache=False,exit_at_step=3)
    for k in range(4):
     h=float(target_lp(m.lm_head(historical[k])[0].float(),target).sum());z=candidate_logprob(m,current['loop_post_norm'][k],target,list(range(len(positions))))
     results.append({'id':row['id'],'candidate':j,'loop':k+1,'historical':h,'capture':z,'absolute_error':abs(h-z)})
 (P.parent/'results'/('HISTORICAL_COMPARE_'+a.device.upper()+'.json')).write_text(json.dumps({'scope':'validation subset, no acceptance gate','device':a.device,'results':results},indent=2)+'\n')
if __name__=='__main__':main()
