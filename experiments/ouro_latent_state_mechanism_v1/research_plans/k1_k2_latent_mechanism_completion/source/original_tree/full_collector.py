"""Sol-owned RTX4090 revision. Fixed original AB1 science, lossless prefix CAS."""
import os,sys,pathlib,json,hashlib,time,fcntl,shutil,argparse,traceback,subprocess
ROOT=pathlib.Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'deps')]
import torch
from ab1_loader_draft import load_model,sha
from ab1_runner_draft import collect_case
from ab1_e2_adapter_draft import apply_frozen_heads
from ab1_prefix_codec import save,load

def write(p,x):
 p=pathlib.Path(p);tmp=p.with_suffix(p.suffix+'.partial');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)

def checkfreeze(f):
 for path,digest in f['sources'].items():assert sha(ROOT/path)==digest,('source drift',path)

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['train64','train1030','validation299'],required=True);a=p.parse_args()
 freeze=ROOT/'FULL_SOURCE_FREEZE.json';f=json.loads(freeze.read_text());checkfreeze(f)
 assert json.loads((ROOT/'technical_three_v2/TECHNICAL_GATE.json').read_text())['status']=='AB1_THREE_TECHNICAL_RECEIPT_AND_INDEPENDENT_REPLAY_GATE_PASSED'
 assert json.loads((ROOT/'CODEC_REAL_TECHNICAL_QA.json').read_text())['status']=='PASS'
 if a.stage!='train64':
  previous='train64' if a.stage=='train1030' else 'train1030'
  assert json.loads((ROOT/'data'/previous/'INDEPENDENT_COVERAGE_AUDIT.json').read_text())['status']=='PASS'
  assert (ROOT/'analysis_results'/previous/'COMPLETE.json').exists(),'Prior stage analysis required'
 split='validation' if a.stage=='validation299' else 'train'
 man=json.loads((ROOT/'inputs/CPU_PREPARATION_MANIFEST.json').read_text());ids=f['cohorts'][a.stage]
 inp=ROOT/'inputs'/(split+'_model_inputs.jsonl');rows={q['id']:q for l in inp.read_text().splitlines() if (q:=json.loads(l))}
 qs=[dict(rows[x],prefix_len=man['prefix_length']) for x in ids];assert len(qs)==len(set(ids))=={'train64':64,'train1030':1030,'validation299':299}[a.stage]
 out=ROOT/'data'/a.stage;out.mkdir(parents=True,exist_ok=True)
 if (out/'PENDING.json').exists():
  done=json.loads((out/'PENDING.json').read_text());assert done['source_freeze_sha256']==sha(freeze) and done['status']=='COMPLETE';print('STAGE_ALREADY_COMPLETE',a.stage);return
 lock=open('/root/autodl-tmp/ouro-e6a/gpu.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert torch.cuda.get_device_name()=='NVIDIA GeForce RTX 4090'
 assert torch.__version__=='2.7.1+cu128' and torch.version.cuda=='12.8'
 uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip();assert uuid==f['gpu_uuid']
 jobs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip().splitlines()
 assert not [x for x in jobs if x.strip() and x.strip()!=str(os.getpid())],'Other GPU compute process present'
 assert torch.cuda.mem_get_info()[0]>20*1024**3 and shutil.disk_usage(ROOT).free>8*1024**3
 torch.set_num_threads(1);torch.set_grad_enabled(False);torch.manual_seed(2021);torch.backends.cuda.matmul.allow_tf32=False
 model,identity=load_model('/root/autodl-tmp/ouro-e4-e5/model',authorized=True,device='cuda:0')
 identity['source_freeze_sha256']=sha(freeze);start=time.monotonic();cases=[]
 for index,q in enumerate(qs):
  if index%16==0:checkfreeze(f);assert shutil.disk_usage(ROOT).free>8*1024**3
  stem=hashlib.sha256(q['id'].encode()).hexdigest();mp=out/(stem+'.json')
  if mp.exists():
   m=json.loads(mp.read_text());assert m['source_identity']==identity and m['input_ids_sha256']==q['input_ids_sha256'];load(m,out)
   cases.append({'id':q['id'],'metadata_file':mp.name,'metadata_sha256':sha(mp),'tensor_raw_sha256':m['raw_tensor_sha256']});continue
  torch.cuda.reset_peak_memory_stats();t=time.monotonic()
  meta,tensors=collect_case(model,q)
  names=[n for n,x in meta['conditions'].items() if x['status']=='OK']
  meta['frozen_e2']=apply_frozen_heads(ROOT/'inputs/selected_models.npz',tensors,q['targets'],names,all_models=True)
  meta.update(split=split,R=q['R'],prefix_len=q['prefix_len'],targets=q['targets'],input_ids_sha256=q['input_ids_sha256'],source_identity=identity,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
  # Upper bound includes every tensor, even an already present shared prefix.
  # Twice raw bytes plus16MiB bounds this codec's gzip/headers/JSON additions.
  upper=2*sum(v.numel()*v.element_size() for v in tensors.values())+16*1024**2
  assert shutil.disk_usage(ROOT).free>8*1024**3+upper,'Insufficient room to commit case while retaining8GiB reserve'
  record=save(out,q['id'],meta,tensors,q['prefix_len']);cases.append(record)
  write(out/'PROGRESS.json',{'status':'RUNNING','stage':a.stage,'completed':len(cases),'n':len(qs),'last_id':q['id'],'elapsed_seconds':time.monotonic()-start,'last_case_seconds':time.monotonic()-t})
  print('AB_FULL_CASE',a.stage,len(cases),len(qs),q['id'],round(time.monotonic()-t,3),flush=True)
 checkfreeze(f);write(out/'PENDING.json',{'status':'COMPLETE','stage':a.stage,'split':split,'n':len(cases),'cases':cases,'source_identity':identity,'source_freeze_sha256':sha(freeze),'labels_sha256':sha(ROOT/'analysis_inputs/analysis_labels.jsonl'),'elapsed_seconds':time.monotonic()-start,'scope':'GPU collection complete; independent full audit and fixed analysis still required.'})
 print('AB_FULL_STAGE_COMPLETE',a.stage,flush=True)

if __name__=='__main__':
 try:main()
 except BaseException as ex:
  write(ROOT/('FAILED_'+str(int(time.time()))+'.json'),{'status':'STOPPED','type':type(ex).__name__,'message':str(ex),'traceback':traceback.format_exc()});raise
