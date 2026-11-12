"""CPU resume preflight; does not create GPU acceptance."""
import json,hashlib,torch
from pathlib import Path
from safetensors.torch import load_file
from full_input import coverage,load
from source_input import ROOT,sha
from decompose import decompose
from state_capture import bits
RUN=ROOT/'research_plans/natural_branch_full_run_20261111';CODE=ROOT/'research_plans/natural_branch_full_code_v2_20261111'
def main():
 qlist=coverage('train');bykey={'train_'+hashlib.sha256(q['id'].encode()).hexdigest():q for q in qlist};metas={p.stem for p in RUN.glob('train_*.json')};tensors={p.stem for p in RUN.glob('train_*.safetensors')}
 if len(metas)!=14 or metas!=tensors or not metas<=set(bykey) or list(RUN.glob('*.partial')):raise ValueError('counts/orphans')
 wp=ROOT/'research_plans/natural_r_geometry_20261111/target_weights.safetensors'
 if sha(wp)!='472f6ab25164a1f2b39fe92432120d6b4176125c086d67bd6174aa95fe8fd647':raise ValueError('weights hash')
 weight=load_file(str(wp));idx={int(x):i for i,x in enumerate(weight['target_ids'])};records=[]
 for key in sorted(metas):
  e=load('train',bykey[key]);mp=RUN/(key+'.json');tp=RUN/(key+'.safetensors');m=json.loads(mp.read_text());t=load_file(str(tp))
  if m['id']!=e['id'] or m['natural_metadata_sha256']!=e['metadata_sha256'] or m['natural_tensor_sha256']!=e['tensor_sha256'] or m['probe_tensor_sha256']!=sha(tp) or m['prior_scores']!=e['scores'] or m['targets']!=e['targets'] or m['gold']!=e['gold'] or m['R_absolute']!=e['R'] or m['R_saved_row']!=e['saved_row']:raise ValueError('identity')
  for k in ('state_capture.py','technical.py','decompose.py','source_input.py','full_input.py','full_runner_draft.py'):
   if m['source_hashes']['code_'+k]!=sha(CODE/k):raise ValueError('code drift')
  states={}
  for u in (0,1):
   for i in range(24):
    d={x:t[f'u{u}_layer{i:02d}_{x}'] for x in ('x','a','y','b','z')}
    if any(v.dtype!=torch.bfloat16 or tuple(v.shape)!=(1,2048) or not torch.isfinite(v).all() for v in d.values()) or not bits(d['x']+d['a'],d['y']) or not bits(d['y']+d['b'],d['z']) or not bits(d['z'],e['states'][(u,i)]):raise ValueError('state')
    if i and not bits(d['x'],states[(u,i-1)]['z']):raise ValueError('continuity')
    if u==1 and i==0 and not bits(d['x'],t['loop1_post_norm']):raise ValueError('loop continuity')
    states[(u,i)]=d
   if not bits(t[f'loop{u+1}_post_norm'],e['norms'][u]) or not bits(t[f'loop{u+1}_logits'],e['logits'][u]):raise ValueError('prior output')
  W=weight['weight_rows'][[idx[x] for x in e['targets']]].float().numpy();gamma=weight['norm_weight'].float().numpy();lens=decompose(states,W,gamma,1e-6,list(range(len(e['targets']))),e['gold'])
  if lens!=m['lens'] or [a['loop_layer'] for a in m['attention']]!=[[u,i] for u in (0,1) for i in range(24)]:raise ValueError('lens/events')
  records.append({'id':e['id'],'metadata_sha256':sha(mp),'tensor_sha256':sha(tp)})
 out=ROOT/'research_plans/natural_branch_preflight14_cpu.json'
 with out.open('x') as f:json.dump({'status':'CPU_STRUCTURE_14_PASS_NOT_CUDA_GATE','count':len(records),'records':records,'script_sha256':sha(Path(__file__))},f,sort_keys=True)
 print('CPU_PREFLIGHT_14_PASS',sha(out),flush=True)
if __name__=='__main__':main()
