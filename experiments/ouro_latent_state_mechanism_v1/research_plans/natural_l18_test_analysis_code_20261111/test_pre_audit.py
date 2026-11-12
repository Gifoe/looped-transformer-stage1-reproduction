"""Test-only CPU pre-audit; independent CUDA replay remains pending."""
import os,json,hashlib
from pathlib import Path
import torch
from safetensors.torch import load_file
import natural_l18_hook,natural_l18_runner,natural_sources,technical,full_input,state_capture,source_input,full_runner
from source_input import ROOT,initial_sources,sha
from natural_sources import load as load_source
from state_capture import bits
if not __debug__:raise RuntimeError('optimized Python forbidden')
RUN=ROOT/'research_plans/natural_l18_full_run_20261111';OUT=ROOT/'research_plans/NATURAL_L18_TEST_PRELIMINARY_COVERAGE_20261111.json';AMEND=ROOT/'research_plans/NATURAL_L18_DISCLOSURE_ORDER_AND_TRAIN_ANALYSIS_AMENDMENT_20261111.md'
def require(x,msg):
 if not x:raise ValueError(msg)
def tsha(t):return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
def main():
 receipt=json.loads((RUN/'receipt.json').read_text());require(receipt['status']=='FULL_L18_COLLECTION_PENDING_INDEPENDENT_COVERAGE' and receipt['counts']=={'train':1119,'validation':299,'test':1172},'full collection incomplete')
 require(sha(AMEND)=='03f5807d4616707804234975dac69cc1157bc0ac503b2b14eda653b7c6d54ed5' and sha(ROOT/'research_plans/NATURAL_L18_TRAIN_INFORMED_PROSPECTIVE_CHECKS_20261111.md')=='6ba60011820c2b2c73c2764af7fb1851b128012606461af11bbdb1fe65bfc31d','disclosure/forecast')
 sources=initial_sources();sources.update({'terms':full_runner.TERMS,'draft':full_runner.DRAFT,'combined':natural_sources.COMBINED,'gpu_replay':natural_sources.GPU,'probe_review':full_runner.REVIEW,'resource_policy':full_runner.POLICY});sh={k:sha(v) for k,v in sources.items()}
 modules=(natural_l18_hook,natural_l18_runner,natural_sources,technical,full_input,state_capture,source_input);paths={'runner':Path(full_runner.__file__).resolve(),**{m.__name__:Path(m.__file__).resolve() for m in modules}};ch={k:sha(v) for k,v in paths.items()}
 qs=full_input.coverage('test');names={'test_'+hashlib.sha256(q['id'].encode()).hexdigest() for q in qs};require(len(names)==1172 and names=={p.stem for p in RUN.glob('test_*.json')}=={p.stem for p in RUN.glob('test_*.safetensors')},'train file IDs');rows=[]
 for q in qs:
  ident=q['id'];src=load_source('test',ident);e=src['natural'];key='test_'+hashlib.sha256(ident.encode()).hexdigest();mp=RUN/(key+'.json');tp=mp.with_suffix('.safetensors');m=json.loads(mp.read_text())
  require(m['id']==ident and m['split']=='test' and m['tensor_sha256']==sha(tp) and m['source_metadata_sha256']==src['source_metadata_sha256'] and m['source_tensor_sha256']==src['source_tensor_sha256'] and m['natural_metadata_sha256']==e['metadata_sha256'] and m['natural_tensor_sha256']==e['tensor_sha256'] and m['combined_sha256']==src['combined_sha256'] and m['input_ids_sha256']==hashlib.sha256(json.dumps(e['ids']).encode()).hexdigest() and m['R']==e['R'] and m['targets']==e['targets'] and m['gold']==e['gold'] and m['source_hashes']==sh and m['code_hashes']==ch,'source')
  require(m['state_sha256']=={str(u):{k:tsha(v) for k,v in src['states'][u].items()} for u in (0,1)},'state hashes')
  controls=m['controls'];require(set(controls)=={'T21_norm','T12_norm'} and all(x.get('status') in ('OK','UNDEFINED_ZERO_DONOR') for x in controls.values()),'control status')
  for name,u in [('T21_norm',0),('T12_norm',1)]:
   is_zero=bool(torch.count_nonzero(src['states'][u]['a'])==0);require((controls[name]['status']=='UNDEFINED_ZERO_DONOR')==is_zero,'control zero status')
   if not is_zero:require(all(torch.isfinite(torch.tensor(controls[name][k])) for k in ('donor_norm','receiver_norm','actual_norm','norm_error')),'nonfinite norm diagnostic')
  expected={'B1','B2','I1','I2','T21','T12'}|{k for k,v in controls.items() if v['status']=='OK'};t=load_file(str(tp));require(set(t)==set(m['scores'])==expected,'keys')
  for k,z in t.items():require(z.dtype==torch.bfloat16 and z.shape==(1,49152) and torch.isfinite(z).all() and len(m['scores'][k])==len(e['targets']) and all(torch.isfinite(torch.tensor(v)) for v in m['scores'][k]),'finite/shape')
  require(bits(t['B1'],e['logits'][0]) and bits(t['B2'],e['logits'][1]) and bits(t['I1'],t['B1']) and bits(t['I2'],t['B2']) and m['scores']['B1']==e['scores'][0] and m['scores']['B2']==e['scores'][1] and m['scores']['I1']==m['scores']['B1'] and m['scores']['I2']==m['scores']['B2'],'baseline/identity')
  if bits(src['states'][0]['a'],src['states'][1]['a']):require(bits(t['T21'],t['B2']) and bits(t['T12'],t['B1']),'equal donor')
  scores=m['scores'];gold=e['gold'];wrong=max((j for j in range(len(e['targets'])) if j!=gold),key=lambda j:scores['B1'][j]);pred={k:max(range(len(v)),key=lambda j:v[j]) for k,v in scores.items()};ties={k:sum(x==max(v) for x in v)>1 for k,v in scores.items()};require(m['fixed_wrong']==wrong and m['predictions']==pred and m['top_ties']==ties,'behavior')
  rows.append({'id':ident,'split':'test','metadata_sha256':sha(mp),'tensor_sha256':sha(tp),'natural_metadata_sha256':e['metadata_sha256'],'natural_tensor_sha256':e['tensor_sha256'],'branch_metadata_sha256':src['source_metadata_sha256'],'branch_tensor_sha256':src['source_tensor_sha256']})
  if len(rows)%100==0:print('TEST_PREAUDITED',len(rows),flush=True)
 require({k:sha(v) for k,v in sources.items()}==sh and {k:sha(v) for k,v in paths.items()}==ch and sha(AMEND)=='03f5807d4616707804234975dac69cc1157bc0ac503b2b14eda653b7c6d54ed5','final drift')
 out={'status':'CPU_SOURCE_BASELINE_IDENTITY_PASS_INDEPENDENT_CUDA_PENDING','count':len(rows),'questions':rows,'sources':sh,'codes':ch,'amendment_sha256':sha(AMEND),'forecast_sha256':sha(ROOT/'research_plans/NATURAL_L18_TRAIN_INFORMED_PROSPECTIVE_CHECKS_20261111.md'),'script_sha256':sha(Path(__file__)),'analysis_script_sha256':sha(Path(__file__).with_name('test_analysis.py')),'limitation':'treatment score metadata comes from original collection; CPU did not independently replay CUDA logsoftmax or norm recipe.'}
 with OUT.open('x') as f:json.dump(out,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
 print('TEST_PREAUDIT_PASS',len(rows),sha(OUT),flush=True)
if __name__=='__main__':main()
