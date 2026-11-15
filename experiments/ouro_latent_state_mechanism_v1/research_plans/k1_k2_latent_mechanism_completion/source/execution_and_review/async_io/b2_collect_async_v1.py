"""Versioned execution-only amendment: one CPU evidence writer, unchanged B2 math."""
import argparse,pathlib,json,hashlib,sys,os,fcntl,subprocess,shutil,time
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text())
def write(p,v):
 with p.open('x') as h:json.dump(v,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
def check(root,f):
 for p,d in f['sources'].items():assert sha(root/p)==d,('source drift',p)
 amendment=read(root/'async_io/EXECUTION_AMENDMENT.json')
 assert amendment['status']=='CPU_ASYNC_STORAGE_ONLY' and amendment['original_science_freeze_sha256']==sha(root/'B2_FULL_SOURCE_FREEZE.json')
 for p,d in amendment['sources'].items():assert sha(root/p)==d,('execution source drift',p)
def audited(d):
 p=d/'PENDING.json';a=read(d/'INDEPENDENT_COVERAGE_AUDIT.json');assert a['status']=='PASS' and a['pending_sha256']==sha(p)
 return a
def resource(root,f,preload=False):
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines();assert len(rows)==1
 name,uuid,free=map(str.strip,rows[0].split(','));assert name=='NVIDIA GeForce RTX 4090' and uuid==f['gpu_uuid'] and (not preload or int(free)>=20480)
 jobs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip().splitlines();assert all(x.strip()==str(os.getpid()) for x in jobs),jobs
 assert shutil.disk_usage(root).free>8*1024**3

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--kind',choices=['reference','treatment'],required=True);p.add_argument('--stage',choices=['train_all','train64','train1030','validation299'],required=True);a=p.parse_args();root=pathlib.Path(a.root).resolve()
 sys.path[:0]=[str(root/'async_io'),str(root/'bridge'),str(root/'code'),str(root/'deps')]
 assert a.kind=='treatment','Reference collector is unchanged and runs separately'
 from async_storage import SingleWriter
 amendment_sha256=sha(root/'async_io/EXECUTION_AMENDMENT.json')
 import torch
 from ab1_loader_draft import load_model
 from ab1_prefix_codec import save,load
 from b2_kernel import reference,treatments
 fp=root/'B2_FULL_SOURCE_FREEZE.json';fh=sha(fp);f=read(fp);assert f['status']=='B2_FINAL25_SCIENCE_FROZEN';check(root,f)
 split='validation' if a.stage=='validation299' else 'train';assert (a.kind=='reference' and a.stage in ('train_all','validation299')) or (a.kind=='treatment' and a.stage!='train_all')
 if split=='validation':
  audited(root/'b2_data/treatments/train1030');an=read(root/'b2_analysis/train1030/COMPLETE.json');assert an['status']=='COMPLETE' and an['audit_sha256']==sha(root/'b2_data/treatments/train1030/INDEPENDENT_COVERAGE_AUDIT.json')
 if a.kind=='treatment':
  audited(root/'b2_data/references'/split)
  if a.stage=='train1030':
   audited(root/'b2_data/treatments/train64')
   an=read(root/'b2_analysis/train64/COMPLETE.json');assert an['status']=='COMPLETE' and an['audit_sha256']==sha(root/'b2_data/treatments/train64/INDEPENDENT_COVERAGE_AUDIT.json')
 ids=f['reference_ids'][split] if a.kind=='reference' else f['cohorts'][a.stage]
 out=root/'b2_data'/('references' if a.kind=='reference' else 'treatments')/(split if a.kind=='reference' else a.stage);out.mkdir(parents=True,exist_ok=True)
 rows={q['id']:q for line in (root/'inputs'/(split+'_model_inputs.jsonl')).read_text().splitlines() if (q:=json.loads(line))};man=read(root/'inputs/CPU_PREPARATION_MANIFEST.json')
 # AB lookup is bound to frozen completed stage manifests; no filename guessing alone.
 lookup={}
 for stage in ('train64','train1030') if split=='train' else ('validation299',):
  dd=root/'data'/stage;audited(dd)
  for c in read(dd/'PENDING.json')['cases']:lookup[c['id']]=(dd,c)
 def ab(ident):
  dd,c=lookup[ident];mp=dd/c['metadata_file'];assert sha(mp)==c['metadata_sha256'];m=read(mp);return mp,m,load(m,dd)
 def ref(ident):
  dd=root/'b2_data/references'/split;rp=read(dd/'PENDING.json');assert rp['source_freeze_sha256']==fh and [c['id'] for c in rp['cases']]==f['reference_ids'][split]
  rc=next(c for c in rp['cases'] if c['id']==ident)
  mp=dd/(hashlib.sha256(ident.encode()).hexdigest()+'.json');m=read(mp);assert sha(mp)==rc['metadata_sha256'] and m['source_freeze_sha256']==fh;return mp,m,load(m,dd)
 lock=open(f['lock_path'],'a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);resource(root,f,preload=True);check(root,f)
 torch.set_num_threads(1);torch.set_grad_enabled(False);torch.manual_seed(2021);torch.backends.cuda.matmul.allow_tf32=False
 model,identity=load_model(f['model_path'],authorized=True,device='cuda:0');producer=dict(identity,source_freeze_sha256=fh);cases=[]
 writer=SingleWriter()
 def record(entry):
  if entry is not None:
   cases.append(entry);print('B2_CASE',a.kind,a.stage,len(cases),len(ids),entry['id'],flush=True)
 for i,ident in enumerate(ids):
  if i%16==0:check(root,f);resource(root,f)
  q=dict(rows[ident],prefix_len=man['prefix_length']);amp,am,at=ab(ident)
  assert identity=={k:v for k,v in am['source_identity'].items() if k!='source_freeze_sha256'}
  mp=out/(hashlib.sha256(ident.encode()).hexdigest()+'.json')
  provenance={'R':q['R'],'prefix_len':q['prefix_len'],'targets':q['targets'],'input_ids_sha256':q['input_ids_sha256'],'AB_metadata_path':str(amp.relative_to(root)),'AB_metadata_sha256':sha(amp),'producer_identity':producer,'source_freeze_sha256':fh}
  if a.kind=='treatment':
   rmp,rm,rt=ref(ident);donor=f['donors'][split][ident];damp,dam,dat=ab(donor);drmp,drm,drt=ref(donor)
   provenance.update(reference_metadata_path=str(rmp.relative_to(root)),reference_metadata_sha256=sha(rmp),donor_id=donor,donor_AB_metadata_path=str(damp.relative_to(root)),donor_AB_metadata_sha256=sha(damp),donor_reference_metadata_path=str(drmp.relative_to(root)),donor_reference_metadata_sha256=sha(drmp))
  if mp.exists():
   record(writer.finish())
   m=read(mp);assert m['id']==ident and all(m[k]==v for k,v in provenance.items());load(m,out)
   if 'execution_amendment_sha256' in m:assert m['execution_amendment_sha256']==amendment_sha256
   record({'id':ident,'metadata_file':mp.name,'metadata_sha256':sha(mp),'tensor_raw_sha256':m['raw_tensor_sha256']})
  else:
   if a.kind=='reference':m,t=reference(model,q,am,at)
   else:m,t=treatments(model,q,am,at,rm,rt,dict(id=donor,split=split,source_identity=dam['source_identity'],tensors=dat),dict(id=donor,split=split,source_identity=drm['source_identity'],tensors=drt))
   m['kernel_status']=m['status'];m['execution_status']='COLLECTED_AWAITING_AUDIT';m.update(provenance);upper=2*sum(v.numel()*v.element_size() for v in t.values())+16*1024**2
   assert shutil.disk_usage(root).free>8*1024**3+2*upper,'two-case archive bound exceeds reserve'
   m['execution_amendment_sha256']=amendment_sha256;m['execution_revision']='async_cpu_writer_v1'
   record(writer.finish())
   writer.submit(out,ident,m,t,q['prefix_len'])
   del t
  del at
  if a.kind=='treatment':del rt,dat,drt
 record(writer.close())
 check(root,f);assert len(cases)==len(ids) and [c['id'] for c in cases]==ids
 pending={'execution_amendment_sha256':amendment_sha256,'status':'COMPLETE','n':len(cases),'split':split,'stage':a.stage,'kind':a.kind,'cases':cases,'source_freeze_sha256':fh,'producer_identity':producer}
 pp=out/'PENDING.json'
 if pp.exists():assert read(pp)==pending
 else:write(pp,pending)
if __name__=='__main__':main()
