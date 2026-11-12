"""Independent CUDA replay and source/metadata audit of completed L18 run."""
import os,json,hashlib,fcntl
from pathlib import Path
import torch
from safetensors.torch import load_file
import natural_l18_hook,natural_l18_runner,natural_sources,technical,full_input,state_capture,source_input,full_runner
from source_input import ROOT,initial_sources,sha
from natural_sources import load as load_source
from natural_l18_hook import matched_norm
from technical import score
from state_capture import bits
if not __debug__:raise RuntimeError('optimized Python forbidden')
RUN=ROOT/'research_plans/natural_l18_full_run_20261111';OUT=ROOT/'research_plans/NATURAL_L18_INDEPENDENT_COVERAGE_20261111.json'
def require(x,msg):
 if not x:raise ValueError(msg)
def tsha(t):return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
def main():
 lock=os.open(ROOT/'ouro_watch.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
 try:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  receipt=json.loads((RUN/'receipt.json').read_text());require(receipt['status']=='FULL_L18_COLLECTION_PENDING_INDEPENDENT_COVERAGE' and receipt['counts']=={'train':1119,'validation':299,'test':1172},'full incomplete')
  sources=initial_sources();sources.update({'terms':full_runner.TERMS,'draft':full_runner.DRAFT,'combined':natural_sources.COMBINED,'gpu_replay':natural_sources.GPU,'probe_review':full_runner.REVIEW,'resource_policy':full_runner.POLICY});sh={k:sha(v) for k,v in sources.items()};require(sh==receipt['source_hashes'],'source hashes')
  modules=(natural_l18_hook,natural_l18_runner,natural_sources,technical,full_input,state_capture,source_input);paths={'runner':Path(full_runner.__file__).resolve(),**{m.__name__:Path(m.__file__).resolve() for m in modules}};ch={k:sha(v) for k,v in paths.items()};require(ch==receipt['code_hashes'] and ch=={'runner':'d3c0efffc52b1b9dd7b09554b222ddf33d3d4e7f3e86e6a40efdc5345a8da5df','natural_l18_hook':'ced59967ba4b40e6866ec84e173f12c659c25ebc2521a37274976a6e1239900d','natural_l18_runner':'8c6537eadfaff8d96bc85ef736176c9b072bd53382599ac59b7e4b7b64d3edc0','natural_sources':'daf71044f6db8e744ab65d4725e8650ccb61b43a7af7694a8cf8ef29088ff362','technical':'537231dbd895221d1775daf31ac6a172f19f97a28cbccb2c066b7e99b20c4dcd','full_input':'0da5cc5db704290daba15d8db23cdd6ee31c636f8adb6b01a23ef0a748717986','state_capture':'f757d947c3eea7966a86f3a17cc5008418488c3e2915ec685cdebb8f30839c1e','source_input':'d52ef07a08011cf28cb78c2883cb86b047b319317a163ce61145fb630963a050'},'code hashes')
  runtime={'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name()};require(runtime==receipt['runtime'],'runtime');rows=[]
  for split in ('train','validation','test'):
   for q in full_input.coverage(split):
    ident=q['id'];src=load_source(split,ident);e=src['natural'];require(all(e['runtime'][k]==runtime[k] for k in runtime) and e['runtime']['attention']=='sdpa' and e['runtime']['dtype']=='bfloat16' and e['runtime']['use_cache'] is False,'natural runtime');key=split+'_'+hashlib.sha256(ident.encode()).hexdigest();mp=RUN/(key+'.json');tp=mp.with_suffix('.safetensors');m=json.loads(mp.read_text())
    require(m['id']==ident and m['split']==split and m['tensor_sha256']==sha(tp) and m['source_metadata_sha256']==src['source_metadata_sha256'] and m['source_tensor_sha256']==src['source_tensor_sha256'] and m['natural_metadata_sha256']==e['metadata_sha256'] and m['natural_tensor_sha256']==e['tensor_sha256'] and m['combined_sha256']==src['combined_sha256'] and m['input_ids_sha256']==hashlib.sha256(json.dumps(e['ids']).encode()).hexdigest() and m['R']==e['R'] and m['targets']==e['targets'] and m['gold']==e['gold'] and m['source_hashes']==sh and m['code_hashes']==ch,'source identity')
    actual_hash={str(u):{k:tsha(v) for k,v in src['states'][u].items()} for u in (0,1)};require(actual_hash==m['state_sha256'],'source states')
    a1=src['states'][0]['a'].to('cuda');a2=src['states'][1]['a'].to('cuda');controls=m['controls'];require(set(controls)=={'T21_norm','T12_norm'},'control names')
    for name,donor,receiver in [('T21_norm',a1,a2),('T12_norm',a2,a1)]:
     value,expected=matched_norm(donor,receiver);require(expected==controls[name],'norm control')
    expected={'B1','B2','I1','I2','T21','T12'}|{k for k,v in controls.items() if v['status']=='OK'};t=load_file(str(tp));require(set(t)==expected==set(m['scores']),'condition keys')
    for k,z in t.items():require(z.dtype==torch.bfloat16 and z.shape==(1,49152) and torch.isfinite(z).all() and score(z.to('cuda'),e['targets'])==m['scores'][k],'score replay')
    require(bits(t['B1'],e['logits'][0]) and bits(t['B2'],e['logits'][1]) and bits(t['I1'],t['B1']) and bits(t['I2'],t['B2']) and m['scores']['B1']==e['scores'][0] and m['scores']['B2']==e['scores'][1],'baseline/identity')
    if bits(a1,a2):require(bits(t['T21'],t['B2']) and bits(t['T12'],t['B1']),'equal donor')
    scores=m['scores'];gold=e['gold'];wrong=max((j for j in range(len(e['targets'])) if j!=gold),key=lambda j:scores['B1'][j]);pred={k:max(range(len(v)),key=lambda j:v[j]) for k,v in scores.items()};ties={k:sum(x==max(v) for x in v)>1 for k,v in scores.items()};require(m['fixed_wrong']==wrong and m['predictions']==pred and m['top_ties']==ties,'prediction/gold/tie')
    rows.append({'id':ident,'split':split,'metadata_sha256':sha(mp),'tensor_sha256':sha(tp),'natural_metadata_sha256':e['metadata_sha256'],'natural_tensor_sha256':e['tensor_sha256'],'branch_metadata_sha256':src['source_metadata_sha256'],'branch_tensor_sha256':src['source_tensor_sha256']})
    if len(rows)%100==0:print('AUDITED',len(rows),flush=True)
  expected_names={r['split']+'_'+hashlib.sha256(r['id'].encode()).hexdigest() for r in rows};actual_json={p.stem for p in RUN.glob('*.json') if p.name!='receipt.json'};actual_tensor={p.stem for p in RUN.glob('*.safetensors')}
  require(not list(RUN.glob('*.partial')) and len(rows)==2590 and len({(r['split'],r['id']) for r in rows})==2590 and expected_names==actual_json==actual_tensor and {k:sha(v) for k,v in sources.items()}==sh and {k:sha(v) for k,v in paths.items()}==ch,'final source/file sets')
  output={'status':'INDEPENDENT_L18_CUDA_AND_SOURCE_COVERAGE_PASSED','count':len(rows),'runtime':runtime,'full_receipt_sha256':sha(RUN/'receipt.json'),'sources':sh,'codes':ch,'audit_script_sha256':sha(Path(__file__)),'questions':rows,'scope':'8-condition saved full logits replay, natural/source state hashes, exact identity, GPU norm control, prediction/tie; injector output is validated by trusted frozen collector hook, not independently recorded as a separate artifact.'}
  with OUT.open('x') as f:json.dump(output,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
  print('L18_INDEPENDENT_COVERAGE_PASS',len(rows),sha(OUT),flush=True)
 finally:os.close(lock)
if __name__=='__main__':main()
