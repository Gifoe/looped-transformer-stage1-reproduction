"""Validation-only preliminary coverage and branch exploration; no CUDA replay."""
import os,sys,json,hashlib,math
from pathlib import Path
import numpy as np,torch
from safetensors.torch import load_file
from safetensors import safe_open
from full_input import coverage,load
from source_input import ROOT,sha,initial_sources
from state_capture import bits
if not __debug__:raise RuntimeError('optimization forbidden')
OUT=ROOT/'research_plans/natural_branch_full_run_20261111'
GEO=ROOT/'research_plans/natural_r_geometry_20261111'
DEST=ROOT/'research_plans/natural_branch_validation_exploration_20261111'
V2='3441948e63da0b1ca02dc59636113715915254a944f32cb46fee0c10ce248d1b'
V3='d19f7fd0b6f1b6b03a73ce821851c1dc4a475b3ffa0138db27ddd19baa11be72'
V4='490ff360f270e67c9516dbeb509c4ed839007261a88485fedc231d5594ddda5b'
V5='70669efd01c2cfd981c31879c0b8b2ebcba5bcd7a81f9378273aa46e8315893d'
CODE_SHA={'state_capture.py':'f757d947c3eea7966a86f3a17cc5008418488c3e2915ec685cdebb8f30839c1e','technical.py':'537231dbd895221d1775daf31ac6a172f19f97a28cbccb2c066b7e99b20c4dcd','decompose.py':'cb730f7b0c09d3e8074a488955ed525d6b50acc29d26b13f23c4405872549243','source_input.py':'d52ef07a08011cf28cb78c2883cb86b047b319317a163ce61145fb630963a050','full_input.py':'0da5cc5db704290daba15d8db23cdd6ee31c636f8adb6b01a23ef0a748717986'}
def independently_recompute(states,W,gamma,eps,gold):
 output=[]
 for u in (0,1):
  for i in range(24):
   v={k:states[(u,i)][k].float().numpy()[0].astype('float64') for k in ('x','a','y','b','z')}
   norms={k:float(np.sqrt(np.mean(v[k]**2)+eps)) for k in ('x','y','z')};z={k:W@(gamma*v[k]/norms[k]) for k in ('x','y','z')}
   def step(a,b):
    sa,sb=norms[a],norms[b];na,nb=z[a]*sa,z[b]*sb;num=.5*(nb-na)*(1/sa+1/sb);den=.5*(na+nb)*(1/sb-1/sa);delta=z[b]-z[a]
    return {'delta':delta.tolist(),'numerator':num.tolist(),'denominator':den.tolist(),'identity_max_error':float(np.max(np.abs(num+den-delta)))}
   output.append({'loop':u+1,'layer':i+1,'attention':step('x','y'),'mlp':step('y','z'),'attention_bf16_addition_residual':(v['y']-v['x']-v['a']).tolist(),'mlp_bf16_addition_residual':(v['z']-v['y']-v['b']).tolist(),'gold':gold})
 return output
def require(x,msg):
 if not x:raise ValueError(msg)
def main():
 require(sha(ROOT/'research_plans/NATURAL_BRANCH_TRAIN_INFORMED_L18_PROSPECTIVE_CHECKS.md')=='492b2ec78065073e1b43eb44daafbb304ffc8096a4e5d11ac68c7653b963408b','prospective freeze drift')
 DEST.mkdir(exist_ok=False)
 source=json.loads((GEO/'source.json').read_text());require(sha(GEO/'validation_rows.jsonl')==source['validation']['rows_sha256'],'geometry rows hash')
 rows=coverage('validation');require(len(rows)==299,'cohort');expected={'validation_'+hashlib.sha256(q['id'].encode()).hexdigest() for q in rows};actual={x.stem for x in OUT.glob('validation_*.json')};require(actual==expected,'validation shards not complete')
 geom={}
 with (GEO/'validation_rows.jsonl').open() as f:
  for line in f:
   g=json.loads(line);require(g['id'] not in geom,'geometry duplicate');geom[g['id']]=g
 require(set(geom)=={q['id'] for q in rows} and not any(x['demo'] for x in geom.values()),'geometry cohort/demo')
 paths=initial_sources();start={k:sha(v) for k,v in paths.items()};config=json.loads(paths['config'].read_text());eps=config['rms_norm_eps']
 with safe_open(str(paths['model']),framework='pt',device='cpu') as f:
  W=f.get_tensor('lm_head.weight').float().numpy().astype('float64');gamma=f.get_tensor('model.norm.weight').float().numpy().astype('float64')
 require(W.shape==(49152,2048) and gamma.shape==(2048,),'model shapes')
 result=[];coverage_rows=[];worst=0.0
 for n,q in enumerate(rows,1):
  e=load('validation',q);g=geom[e['id']];key='validation_'+hashlib.sha256(e['id'].encode()).hexdigest();mp=OUT/(key+'.json');tp=OUT/(key+'.safetensors');m=json.loads(mp.read_text())
  require(m['id']==e['id'] and m['split']=='validation' and m['natural_metadata_sha256']==e['metadata_sha256'] and m['natural_tensor_sha256']==e['tensor_sha256'],'source')
  require(m['probe_tensor_sha256']==sha(tp) and m['targets']==e['targets'] and m['gold']==e['gold'] and m['R_absolute']==e['R'] and m['R_saved_row']==e['saved_row'] and m['prior_scores']==e['scores'],'metadata')
  require(g['metadata_sha256']==e['metadata_sha256'] and g['tensor_sha256']==e['tensor_sha256'] and g['targets']==e['targets'] and g['gold']==e['gold'] and g['scores']==e['scores'],'geometry identity')
  p1=int(np.argmax(e['scores'][0]));p2=int(np.argmax(e['scores'][1]));gold=e['gold'];score0=e['scores'][0];wr=max((j for j in range(len(score0)) if j!=gold),key=lambda j:score0[j]);group=('stable_correct' if p2==gold else 'damage') if p1==gold else ('rescue' if p2==gold else 'stable_wrong');ties=[sum(v==max(sc) for v in sc)>1 for sc in e['scores']]
  require(g['predictions']==[p1,p2] and g['top_ties']==ties and g['fixed_wrong']==wr and g['group']==group and g['natural_no_top_tie']==(not any(ties)),'natural behaviour')
  sh=m['source_hashes'];require(sh['code_full_runner_draft.py'] in (V2,V3,V4,V5) and all(sh[k]==v for k,v in start.items()) and all(sh['code_'+k]==v for k,v in CODE_SHA.items()),'fingerprint')
  t=load_file(str(tp));names={f'u{u}_layer{i:02d}_{k}' for u in (0,1) for i in range(24) for k in ('x','a','y','b','z')}|{f'loop{u}_logits' for u in (1,2)}|{f'loop{u}_post_norm' for u in (1,2)};require(set(t)==names,'tensor keys')
  att=m['attention'];require(len(att)==48 and [x.get('loop_layer') for x in att]==[[u,i] for u in (0,1) for i in range(24)],'attention events')
  for x in att:
   require(x['past_is_none'] is True and x['use_cache'] is False and set(('attention_mask','position_ids','cache_position','rope','backend_kernel'))<=set(x) and isinstance(x['rope'],list) and len(x['rope'])==2 and all(x['rope']),'attention fields')
  states={}
  for u in (0,1):
   for i in range(24):
    d={k:t[f'u{u}_layer{i:02d}_{k}'] for k in ('x','a','y','b','z')};require(all(v.dtype==torch.bfloat16 and tuple(v.shape)==(1,2048) and torch.isfinite(v).all() for v in d.values()),'state shape/finiteness')
    require(bits(d['x']+d['a'],d['y']) and bits(d['y']+d['b'],d['z']) and bits(d['z'],e['states'][(u,i)]),'residual / old layer')
    if i:require(bits(d['x'],states[(u,i-1)]['z']),'layer continuity')
    if u==1 and i==0:require(bits(d['x'],t['loop1_post_norm']),'loop continuity')
    states[(u,i)]=d
   for suffix,prior,shape in [('post_norm',e['norms'][u],(1,2048)),('logits',e['logits'][u],(1,49152))]:
    v=t[f'loop{u+1}_{suffix}'];require(v.dtype==torch.bfloat16 and tuple(v.shape)==shape and torch.isfinite(v).all() and bits(v,prior),'norm/logits anchor')
  weights=W[e['targets']];lens=independently_recompute(states,weights,gamma,eps,e['gold']);require(lens==m['lens'],'recompute lens')
  gold=e['gold'];wrong=g['fixed_wrong'];others=[j for j in range(len(e['targets'])) if j!=gold]
  def gaps(values):return [float(values[gold]-values[j]) for j in others]
  def candidate_metric(values):return {'all_wrong':float(np.mean(gaps(values))),'fixed_wrong':float(values[gold]-values[wrong])}
  layers=[]
  for i in range(24):
   a1=lens[i]['attention'];a2=lens[i+24]['attention'];b1=lens[i]['mlp'];b2=lens[i+24]['mlp'];item={'layer':i+1}
   for name,s1,s2 in [('attention',a1,a2),('mlp',b1,b2)]:
    for ss in (s1,s2):require(np.allclose(np.asarray(ss['numerator'])+np.asarray(ss['denominator']),ss['delta'],rtol=1e-10,atol=1e-10),'segment decomposition')
    item[name+'_candidate_values']={'k1':{k:s1[k] for k in ('delta','numerator','denominator')},'k2':{k:s2[k] for k in ('delta','numerator','denominator')}}
    for component in ('delta','numerator','denominator'):
     x=candidate_metric(np.array(s1[component]));y=candidate_metric(np.array(s2[component]));item[name+'_'+component]={'k1':x,'k2':y,'difference':{k:y[k]-x[k] for k in x}}
   for label,le,ri,branch in [('attention','x','y','a'),('mlp','y','z','b')]:
    parts=[]
    for u in (0,1):
     d=states[(u,i)];v={k:d[k].float().numpy()[0].astype('float64') for k in (le,ri,branch)}
     factor=.5*(1/math.sqrt(np.mean(v[le]*v[le])+eps)+1/math.sqrt(np.mean(v[ri]*v[ri])+eps))
     branch_vector=weights@(gamma*v[branch]);round_vector=weights@(gamma*(v[ri]-v[le]-v[branch]))
     ref=lens[i+24*u][label]['numerator'];require(np.allclose(factor*(branch_vector+round_vector),ref,rtol=1e-10,atol=1e-10),'branch numerator accounting')
     parts.append({'branch':candidate_metric(factor*branch_vector),'rounding':candidate_metric(factor*round_vector)})
    item[label+'_numerator_parts']=parts
   layers.append(item)
  def lens_gap(u,i,k):
   v=states[(u,i)][k].float().numpy()[0].astype('float64');z=weights@(gamma*v/math.sqrt(np.mean(v*v)+eps));return candidate_metric(z)
  endpoints={(u,k):lens_gap(u,0 if k=='x' else 23,k) for u in (0,1) for k in ('x','z')}
  for j in range(len(e['targets'])):
   def lv(u,i,k):
    v=states[(u,i)][k].float().numpy()[0].astype('float64');return weights@(gamma*v/math.sqrt(np.mean(v*v)+eps))
   total=(lv(1,23,'z')[j]-lv(0,23,'z')[j])-(lv(1,0,'x')[j]-lv(0,0,'x')[j]);layer_sum=sum(lens[i+24]['attention']['delta'][j]+lens[i+24]['mlp']['delta'][j]-lens[i]['attention']['delta'][j]-lens[i]['mlp']['delta'][j] for i in range(24));require(np.isclose(total,layer_sum,rtol=1e-10,atol=1e-10),'candidate telescope')
  for name in ('all_wrong','fixed_wrong'):
   startdiff=endpoints[(1,'x')][name]-endpoints[(0,'x')][name];enddiff=endpoints[(1,'z')][name]-endpoints[(0,'z')][name];sm=sum(x['attention_delta']['difference'][name]+x['mlp_delta']['difference'][name] for x in layers);worst=max(worst,abs(enddiff-startdiff-sm));require(np.isclose(enddiff-startdiff,sm,rtol=1e-10,atol=1e-10),'gap telescope')
  result.append({'id':e['id'],'demo':g['demo'],'group':g['group'],'natural_no_top_tie':g['natural_no_top_tie'],'gold':gold,'fixed_wrong':wrong,'targets':e['targets'],'natural_scores':e['scores'],'layers':layers,'telescoping':{'initial_k1':endpoints[(0,'x')],'initial_k2':endpoints[(1,'x')],'final_k1':endpoints[(0,'z')],'final_k2':endpoints[(1,'z')]}})
  coverage_rows.append({'id':e['id'],'natural_metadata_sha256':e['metadata_sha256'],'natural_tensor_sha256':e['tensor_sha256'],'branch_metadata_sha256':sha(mp),'branch_tensor_sha256':sha(tp)})
  if n%50==0:print('AUDITED',n,flush=True)
 require({k:sha(v) for k,v in paths.items()}==start,'source changed')
 def write(name,obj):
  with (DEST/name).open('x') as f:json.dump(obj,f,sort_keys=True,allow_nan=False);f.write('\n')
 write('validation_rows.json',result);write('validation_coverage_preliminary.json',{'status':'PRELIMINARY_CPU_ANCHORS_PASS_CUDA_REPLAY_PENDING','count':len(result),'source':start,'natural_coverage_sha256':source['validation']['coverage_sha256'],'geometry_rows_sha256':source['validation']['rows_sha256'],'rows_sha256':sha(DEST/'validation_rows.json'),'disclosure_sha256':sha(ROOT/'research_plans/NATURAL_BRANCH_TRAIN_INFORMED_L18_PROSPECTIVE_CHECKS.md'),'script_sha256':sha(Path(__file__)),'summary_script_sha256':sha(ROOT/'research_plans/natural_branch_validation_analysis_code_20261111/summarize_validation.py'),'max_telescope_error':worst,'questions':coverage_rows})
 print('VALIDATION_PRELIMINARY_COVERAGE_COMPLETE',len(result),'telescope_error',worst,flush=True)
if __name__=='__main__':main()
