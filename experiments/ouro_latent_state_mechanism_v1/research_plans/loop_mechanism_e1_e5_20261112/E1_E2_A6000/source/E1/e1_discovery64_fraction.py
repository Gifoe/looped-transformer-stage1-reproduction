"""Exact BF16 target-logit sensitivity for audited pre-effect discovery64; no new GPU."""
import json,hashlib,os,datetime
from pathlib import Path
from fractions import Fraction
from safetensors.torch import load_file
B=Path('/home/luansiyu/loop-latent/repo/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112');E=B/'E1';RUN=E/'train';OLD=Path('/home/luansiyu/loop-latent/research_plans/natural_qkv_eight_corner_20261112');A=E/'discovery64_cuda_audit.json';SEL=B/'orders/E1_DISCOVERY64_FROZEN.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
if not __debug__:raise RuntimeError('optimized Python forbidden')
a=json.loads(A.read_text());sel=json.loads(SEL.read_text());assert a['status']=='DISCOVERY64_INDEPENDENT_CUDA_SCORE_AND_SOURCE_AUDIT_PASSED' and a['count']==64 and a['selection_sha256']==sha(SEL);groups={v['id']:g for g,vs in sel['groups'].items() for v in vs};assert len(groups)==64
rows=[]
for z in a['questions']:
 ident=z['id'];key=hashlib.sha256(ident.encode()).hexdigest();mp=RUN/(key+'.json');tp=mp.with_suffix('.safetensors');assert sha(mp)==z['metadata_sha256'] and sha(tp)==z['tensor_sha256'];m=json.loads(mp.read_text());t=load_file(str(tp));om=OLD/'train'/(key+'.json');op=om.with_suffix('.safetensors');assert sha(om)==z['old_metadata_sha256'] and sha(op)==z['old_tensor_sha256'];o=json.loads(om.read_text());ot=load_file(str(op));gold=o['gold'];targets=o['targets'];wrong=[j for j in range(len(targets)) if j!=gold];fixed=o['fixed_wrong'];entry={'id':ident,'group':groups[ident],'q':{}}
 for q in (1,2):
  c=m['controls']['conditions'][str(q)]
  def raw(z):return [Fraction.from_float(float(z[0,j])) for j in targets]
  def gaps(vals):return [vals[gold]-vals[j] for j in wrong]
  f={str(j)+str(k):gaps(raw(ot[f'G{q}{j}{k}_logits'])) for j,k in ((1,1),(1,2),(2,1),(2,2))};f['ADD']=gaps(raw(t[f'q{q}_ADD_logits']));f['UNRELATED']=gaps(raw(t[f'q{q}_UNRELATED_logits']));f['ADD_NORM']=gaps(raw(t[f'q{q}_ADD_NORM_logits'])) if 'ADD_NORM' in c else None
  def diff(k):
   if k=='removed':return [a-b for a,b in zip(f['22'],f['ADD'])]
   if k=='suffix':return [v-a-b+d for v,a,b,d in zip(f['ADD'],f['21'],f['12'],f['11'])]
   if k=='total':return [v-a-b+d for v,a,b,d in zip(f['22'],f['21'],f['12'],f['11'])]
   if k=='paired_unrelated':return [u-v for u,v in zip(f['UNRELATED'],f['ADD'])]
   if k=='paired_norm':return [u-v for u,v in zip(f['ADD_NORM'],f['ADD'])] if f['ADD_NORM'] is not None else None
  result={}
  for name in ('removed','suffix','total','paired_unrelated','paired_norm'):
   d=diff(name)
   if d is None:result[name]=None;continue
   allmean=sum(d,Fraction())/len(d);fixedval=d[wrong.index(fixed)];result[name]={'allwrong_sign':(allmean>0)-(allmean<0),'fixedwrong_sign':(fixedval>0)-(fixedval<0),'allwrong_exact':str(allmean),'fixedwrong_exact':str(fixedval)}
  entry['q'][str(q)]=result
 rows.append(entry)
assert len(rows)==64 and len({x['id'] for x in rows})==64
out={'status':'DISCOVERY64_FRACTION_RAW_LOGIT_SENSITIVITY_COMPLETE','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'audit_sha256':sha(A),'selection_sha256':sha(SEL),'script_sha256':sha(Path(__file__)),'scope':'Exact target BF16 raw-logit differences; production scores remain unchanged. Raw exact zero is not strict positive or negative. Groups are 16 each, not representative cohort weights.','rows':rows,'groups':{}}
for g in sorted(set(groups.values())):
 out['groups'][g]={}
 for q in ('1','2'):
  out['groups'][g][q]={}
  for name in ('removed','suffix','total','paired_unrelated','paired_norm'):
   values=[(r['id'],r['q'][q][name]) for r in rows if r['group']==g and r['q'][q][name] is not None];out['groups'][g][q][name]={axis:{'positive_ids':sorted(i for i,v in values if v[axis]>0),'negative_ids':sorted(i for i,v in values if v[axis]<0),'zero_ids':sorted(i for i,v in values if v[axis]==0)} for axis in ('allwrong_sign','fixedwrong_sign')}
p=E/'discovery64_analysis/fraction_sensitivity.json'
with p.open('x') as f:json.dump(out,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
print('DISCOVERY64_FRACTION_COMPLETE',sha(p),flush=True)
