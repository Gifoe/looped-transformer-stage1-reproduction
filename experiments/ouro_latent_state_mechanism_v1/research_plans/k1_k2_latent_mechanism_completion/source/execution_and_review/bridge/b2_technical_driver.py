"""Candidate technical-only driver. No launch on import. Technical cyclic donors only."""
import argparse,pathlib,sys,json,hashlib,os,time,subprocess,shutil,datetime

def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def independent_control(base,target,direction,kind,sign):
 import torch
 b=base.double();t=target.double();scale=torch.ones((),dtype=torch.float64)
 if kind=='norm':
  if b.norm()==0 and t.norm()!=0:raise ValueError('undefined zero norm base')
  if b.norm()!=0:scale=t.norm()/b.norm()
  exact=b*scale
 else:
  dose=(t-b).norm();dn=direction.double().norm()
  if dose==0:scale=scale*0;exact=b.clone()
  elif dn==0:raise ValueError('undefined zero donor direction')
  else:scale=dose/dn;exact=b+sign*scale*direction.double()
 after=exact.to(base.dtype);return after,dict(control_before=base,control_exact_before_cast=exact,control_after=after,control_cast_residual=after.double()-exact,control_scale_fp64=scale)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--tag',default='b2_technical');args=ap.parse_args();assert args.tag.replace('_','').replace('-','').isalnum();root=pathlib.Path(args.root);sys.path[:0]=[str(root/x) for x in ('code','deps','bridge')]
 import torch,fcntl
 from ab1_prefix_codec import save,load
 from ab1_loader_draft import load_model
 from b2_kernel import reference,treatments
 from b2_independent import replay
 def same(a,b):assert a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().reshape(-1).view(torch.uint8),b.contiguous().reshape(-1).view(torch.uint8))
 freeze=root/'FULL_SOURCE_FREEZE.json';f=json.loads(freeze.read_text());techfreeze=root/'B2_TECH_SOURCE_FREEZE.json';tf=json.loads(techfreeze.read_text());sources=tf['sources'];assert set(f['sources'])<=set(sources);assert all(sources[k]==v for k,v in f['sources'].items());assert all(str(x.relative_to(root)) in sources for x in (root/'bridge').glob('*.py'));assert {'FULL_SOURCE_FREEZE.json','A_SIDE_SOURCE_FREEZE.json','pipeline/describe_A.py','pipeline/describe_A_vectors.py','data/train64/PENDING.json','data/train64/INDEPENDENT_COVERAGE_AUDIT.json'}<=set(sources);before={k:sha(root/k) for k in sources};assert before==sources;technical_ids=tf['technical_train_ids'];assert tf['technical_rule']==json.loads((root/'inputs/CPU_PREPARATION_MANIFEST.json').read_text())['technical_rule'];out=root/args.tag;assert not out.exists() or not any(out.iterdir());assert shutil.disk_usage(root).free>=8*1024**3;out.mkdir(exist_ok=True);started=dict(status='STARTED',tag=args.tag,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_freeze_sha256=sha(techfreeze),sources=before,runtime_python=sys.version);(out/'STARTED.json').write_text(json.dumps(started,indent=2)+'\n');assert json.loads((root/'QUEUE_STATE.json').read_text())['status']=='COMPLETE'
 for stage in ('train64','train1030','validation299'):assert json.loads((root/'analysis_results'/stage/'COMPLETE.json').read_text())['status']=='COMPLETE'
 lock=open('/root/autodl-tmp/ouro-e6a/gpu.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert torch.cuda.get_device_name()=='NVIDIA GeForce RTX 4090' and torch.__version__=='2.7.1+cu128' and torch.version.cuda=='12.8';assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==f['gpu_uuid'];jobs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines();assert not [x for x in jobs if x.strip() and x.strip()!=str(os.getpid())];assert torch.cuda.mem_get_info()[0]>20*1024**3
 torch.set_num_threads(1);torch.manual_seed(2021);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;model,identity=load_model('/root/autodl-tmp/ouro-e4-e5/model',authorized=True,device='cuda:0');torch.cuda.reset_peak_memory_stats();start=time.monotonic();d=root/'data/train64';p=json.loads((d/'PENDING.json').read_text());audit=json.loads((d/'INDEPENDENT_COVERAGE_AUDIT.json').read_text());assert audit['status']=='PASS' and audit['pending_sha256']==sha(d/'PENDING.json');manifest=json.loads((root/'inputs/CPU_PREPARATION_MANIFEST.json').read_text());ids=manifest['technical_train_ids'];assert ids==technical_ids;assert len(ids)==3 and len(set(ids))==3;inputs={q['id']:q for line in (root/'inputs/train_model_inputs.jsonl').read_text().splitlines() if (q:=json.loads(line))};cases={x['id']:x for x in p['cases']};data={};records=[];actual_production_calls=0;actual_manual_calls=0;production_partial_attempts=0
 for ident in ids:
  c=cases[ident];mp=d/c['metadata_file'];assert sha(mp)==c['metadata_sha256'];m=json.loads(mp.read_text());assert {k:v for k,v in identity.items() if k!='source_freeze_sha256'}=={k:v for k,v in m['source_identity'].items() if k!='source_freeze_sha256'};t=load(m,d);q=dict(inputs[ident],prefix_len=manifest['prefix_length']);tm,tt=reference(model,q,m,t);actual_production_calls+=3;data[ident]=(q,m,t,tm,tt)
 for index,ident in enumerate(ids):
  q,m,ab,tm,tt=data[ident];donor=ids[(index+1)%3];dq,dm,dab,dtm,dt=data[donor];cm,ct=treatments(model,q,m,ab,tm,tt,dict(id=donor,split=q['split'],source_identity=dm['source_identity'],tensors=dab),dict(id=donor,split=q['split'],source_identity=dm['source_identity'],tensors=dt));actual_production_calls+=cm['work_counts']['body_forward_attempts'];expected_keys=set(['TX','TA','TXA','U_plus','upstream_norm','upstream_unrelated','U_plus_norm_preserve','U_plus_unrelated','M','U_M_plus','M_norm_preserve','M_unrelated','U_M_plus_norm_preserve','U_M_plus_unrelated']+[f'{n}_{b}_{k}' for n in ('T','N') for b in ('X','A') for k in ('norm','unrelated')]);assert set(cm['conditions'])==expected_keys;tensors={**tt,**ct};reserve=sum(v.numel()*v.element_size() for v in tensors.values())*(len(ids)-index);assert shutil.disk_usage(root).free>=8*1024**3+reserve;E=ab['source_E'];B=ab['source_B'];U=B.clone();U[:,q['prefix_len']:q['R']]=E[:,q['prefix_len']:q['R']];delta={i:ab[f'H11_L{i+1:02d}_a_R'].double()-ab[f'H01_L{i+1:02d}_a_R'].double() for i in range(9,16)};configs={'N':{},'U':dict(boundary=U),'T':dict(delta=delta),'TX':dict(delta=delta,x_branch=ab['H11_L18_x_R']),'TA':dict(delta=delta,a_branch=ab['H11_L18_a_R']),'TXA':dict(delta=delta,x_branch=ab['H11_L18_x_R'],a_branch=ab['H11_L18_a_R']),'U_plus':dict(boundary=U,delta=delta,sign=1),'upstream_norm':dict(delta=delta,norm_preserve=True),'U_plus_norm_preserve':dict(boundary=U,delta=delta,sign=1,norm_preserve=True)}
  md={i:tt[f'd_mlp_{i:02d}'] for i in range(9,16)}
  configs.update(M=dict(mlp_delta=md),U_M_plus=dict(boundary=U,mlp_delta=md,mlp_sign=1),M_norm_preserve=dict(mlp_delta=md,mlp_norm_preserve=True),U_M_plus_norm_preserve=dict(boundary=U,mlp_delta=md,mlp_sign=1,mlp_norm_preserve=True))
  for mn,kwargs in [('M_unrelated',{}),('U_M_plus_unrelated',dict(boundary=U,mlp_sign=1))]:
   if cm['conditions'].get(mn,{}).get('status')=='OK':
    ud={}
    for i in range(9,16):
     direction=dt[f'd_mlp_{i:02d}'];dose=md[i].norm();dn=direction.norm();assert dose==0 or dn!=0;ud[i]=torch.zeros_like(direction) if dose==0 else dose*direction/dn
    configs[mn]=dict(mlp_delta=ud,**kwargs)
  if any(cm['conditions'].get(n,{}).get('status')=='OK' for n in ('upstream_unrelated','U_plus_unrelated')):
   ud={}
   for i in range(9,16):
    dd=dab[f'H11_L{i+1:02d}_a_R'].double()-dab[f'H01_L{i+1:02d}_a_R'].double();ud[i]=torch.zeros_like(dd) if delta[i].norm()==0 else delta[i].norm()*dd/dd.norm();same(ud[i],ct[f'upstream_unrelated_control_requested_delta_{i:02d}'])
   
   if cm['conditions']['upstream_unrelated']['status']=='OK':configs['upstream_unrelated']=dict(delta=ud)
   if cm['conditions']['U_plus_unrelated']['status']=='OK':configs['U_plus_unrelated']=dict(boundary=U,delta=ud,sign=1)
  for branch,key in [('X','x'),('A','a')]:
   N=ab[f'H11_L18_{key}_R'];T=tt[f'T_branch_{key}_actual_R'];direction=dab[f'H11_L18_{key}_R'].double()-dt[f'T_branch_{key}_actual_R'].double()
   for label,base,target,sign in [('T',T,N,1),('N',N,T,-1)]:
    for kind in ('norm','unrelated'):
     name=f'{label}_{branch}_{kind}'
     if cm['conditions'][name]['status']!='OK':continue
     value,receipt=independent_control(base,target,direction,kind,sign)
     for k,v in receipt.items():same(v,ct[name+'_'+k])
     configs[name]={**(dict(delta=delta) if label=='T' else {}),key+'_branch':value}
  compared=[]
  for name,config in configs.items():
   if name not in ('N','U','T') and cm['conditions'][name]['status']!='OK':continue
   r=replay(model,q,**config);actual_manual_calls+=1;same(r['E'],E);same(r['B'],B);prefix={'N':'H11','U':'H01'}.get(name,name);expected=ab if name in ('N','U') else tensors;same(r['logits'],expected[prefix+'_logits'])
   if name in ('N','U'):
    for i in range(9,16):
     same(r['sites'][i]['before'],ab[f'{prefix}_L{i+1:02d}_a_R']);same(r['sites'][i]['mlp_before'],tt[f'{name}_mlp_{i:02d}'])
    for key,field in [('actual_x_R','L18_x_R'),('actual_a_R','L18_a_R'),('used_y_R','L18_y_R'),('finalnorm_R','finalnorm_R')]:same(r['branch'][key],ab[prefix+'_'+field])
    q9=ab['native_Q9_full' if name=='N' else 'U_Q9_full'];same(r['Q9_actual'],q9)
   if name not in ('N','U'):
    for i,site in r['sites'].items():
     for k,v in site.items():
      stored=f'{name}_site_{i:02d}_{k}'
      assert stored in tensors;same(v,tensors[stored])
    mapping={'actual_x_R':'x_actual_R','actual_a_R':'a_actual_R','used_y_R':'y_branch_R','used_x_R':'x_branch_used_R','used_a_R':'a_branch_used_R','MLP_skip_R':'MLP_skip_shared_alias_R','b_R':'b_actual_R','postMLP_R':'after_mlp_R','finalnorm_R':'final_norm_R'}
    for k,target in mapping.items():
     stored=name+'_branch_'+target
     assert stored in tensors;same(r['branch'][k],tensors[stored])
    metadata=tm['conditions']['T'] if name=='T' else cm['conditions'][name];assert {str(k):v for k,v in r['nonr'].items()}=={str(k):v for k,v in metadata['nonR_trace'].items()}
   compared.append(name)
  assert set(compared)=={'N','U','T'}|{k for k,v in cm['conditions'].items() if v['status']=='OK'};meta=dict(tag=args.tag,id=ident,split=q['split'],R=q['R'],prefix_len=q['prefix_len'],reference=tm,treatments=cm,independent_replayed=compared,expected_endpoints=25,actual_replayed_count=len(compared),undefined_endpoints={k:v for k,v in cm['conditions'].items() if v['status']!='OK'},donor_id=donor,donor_scope='TECHNICAL_ONLY cyclic next of fixed three, not scientific donor mapping',AB_metadata_sha256=cases[ident]['metadata_sha256']);records.append(save(out,ident,meta,tensors,q['prefix_len']))
 torch.cuda.synchronize();assert all(sha(root/k)==v for k,v in before.items()) and sha(techfreeze)==started['source_freeze_sha256'];gate=dict(status='B2_THREE_TECHNICAL_AVAILABLE_ENDPOINT_WIRING_PASS',tag=args.tag,technical_freeze_sha256=sha(techfreeze),cases=records,source_sha256=before,identity=identity,peak_allocated_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.monotonic()-start,production_body_forward_attempts=actual_production_calls,production_complete_body_forwards=actual_production_calls,production_partial_forward_attempts=production_partial_attempts,production_total_forward_attempts=actual_production_calls+production_partial_attempts,independent_body_equivalents=actual_manual_calls,scope='Technical wiring only. Undefined controls retained; no scientific result or full B2 experiment authorization.');(out/'TECHNICAL_GATE.json').write_text(json.dumps(gate,indent=2)+'\n')
if __name__=='__main__':main()
