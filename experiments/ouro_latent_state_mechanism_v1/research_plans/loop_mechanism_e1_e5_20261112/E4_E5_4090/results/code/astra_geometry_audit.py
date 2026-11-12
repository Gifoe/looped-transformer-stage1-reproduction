"""Read saved compact tensors independently; descriptive geometry and control norms."""
import argparse,json,hashlib,statistics,collections
from pathlib import Path
import torch
from safetensors.torch import load_file
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def thash(t):return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.results)
 files=list(root.glob('*.native.json'));byid={read(p)['id']:p for p in files};geoms=[];controls=[];fail=[];buckets=collections.defaultdict(list)
 for np in files:
  m=read(np);t=load_file(str(np.with_suffix('.safetensors')))
  for lo in (1,2,3):
   hi=lo+1
   for state in ['O','a_R','x_R','final_norm_R']:
    x=t[f'K{lo}_{state}'].double().flatten();y=t[f'K{hi}_{state}'].double().flatten();dx=float((y-x).norm());nx=float(x.norm());ny=float(y.norm())
    r={'id':m['id'],'split':m['split'],'pair':f'{lo}-{hi}','state':state,'delta_l2':dx,'relative_delta_l2':dx/nx if nx else None,'earlier_norm':nx,'later_norm':ny,'cosine':float(x.dot(y)/(x.norm()*y.norm())) if nx and ny else None}
    geoms.append(r);buckets[(m['split'],r['pair'],state)].append(r)
  rp=np.with_name(np.name.replace('.native.json','.results.json'))
  if not rp.exists():continue
  for r in read(rp)['rows']:
   if r['condition'] not in ['target','norm_only','unrelated_delta']:continue
   i,j=r['receiver_loop'],r['donor_loop'];base=t[f'K{i}_O'];target=t[f'{i}_{j}_target'];d=target.float()-base.float()
   if r['native_sdpa_R_sha256']!=thash(base):fail.append([m['id'],i,j,'native_hash'])
   if r['condition']=='target' and r['injected_sha256']!=thash(target):fail.append([m['id'],i,j,'target_hash'])
   z={'id':m['id'],'pair':r['pair'],'direction':r['direction'],'condition':r['condition'],'target_delta_norm_cpu':float(d.norm()),'actual_delta_norm_recorded':r['actual_delta_norm'],'endpoint_norm_recorded':r['injected_norm']}
   if r['condition']=='unrelated_delta':
    ot=load_file(str(byid[r['donor_id']].with_suffix('.safetensors')));od=ot[f'{i}_{j}_target'].float()-ot[f'K{i}_O'].float();pre=d.norm()*od/od.norm();patch=(base.float()+pre).to(base.dtype)
    z.update({'unrelated_source_delta_norm':float(od.norm()),'scaled_delta_norm_before_cast':float(pre.norm()),'delta_norm_after_cast_cpu':float((patch.float()-base.float()).norm()),'recorded_to_target_delta_norm_ratio':r['actual_delta_norm']/float(d.norm()) if float(d.norm()) else None})
   if r['condition']=='norm_only':z['recorded_to_target_endpoint_norm_ratio']=r['injected_norm']/float(target.float().norm())
   controls.append(z)
 summary=[]
 for k,rows in sorted(buckets.items()):
  s=dict(zip(['split','pair','state'],k));s['n']=len(rows)
  for field in ['delta_l2','relative_delta_l2','earlier_norm','later_norm','cosine']:
   vals=[r[field] for r in rows if r[field] is not None];s[field]={'mean':statistics.mean(vals),'median':statistics.median(vals),'min':min(vals),'max':max(vals)}
  summary.append(s)
 result={'status':'FAILED' if fail else 'DESCRIPTIVE_TENSORS_AUDITED','n':len(files),'failures':fail,'summary':summary,'per_question_geometry':geoms,'control_norms':controls,'scope':'CPU FP64 native geometry; CPU FP32 control-norm reconstruction; no claim of full-model FP32 or hidden-state convergence'}
 Path(a.output).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({'status':result['status'],'n':len(files),'failures':fail}))
 if fail:raise SystemExit(1)
if __name__=='__main__':main()
