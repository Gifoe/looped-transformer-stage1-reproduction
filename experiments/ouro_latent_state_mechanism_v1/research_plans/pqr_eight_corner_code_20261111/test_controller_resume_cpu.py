"""Exercise real PQR controller control flow with synthetic CPU stage executor."""
import json,sys,pathlib,tempfile,os,uuid,hashlib,signal,torch
from safetensors.torch import save_file
P=pathlib.Path('/home/luansiyu/loop-latent');sys.path.insert(0,str(P/'repo/experiments/ouro_latent_state_mechanism_v1'))
import pqr_controller as c

def write(p,obj):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj))
def opt(args,key):return pathlib.Path(args[args.index(key)+1])
def run_test(root):
 c.ROOT=root;c.run=root/'orchestration_runs/pqr-round3-20261111';c.HERE=root/'draft';c.HERE.mkdir();old=root/'orchestration_runs/round2-resume-20261111/status.json';write(old,{'status':'INTERVENTION_COMPLETE_PENDING_ASTRA_REVIEW'});state={'pass':0,'calls':[]}
 c.external_tasks=lambda:[];c.machine=lambda:{'gpu':{'free_mib':40000,'compute_pids':[]},'disk_free_bytes':200*c.GIB,'memory':{'MemAvailable':40*c.GIB}};c.time.sleep=lambda x:None
 def fake(args,log,cwd,env,monitor,period=5):
  with log.open('x') as f:f.write('synthetic stage\n')
  cmd=pathlib.Path(args[2]).name;state['calls'].append((state['pass'],cmd,args))
  if cmd=='pqr_binding.py':write(opt(args,'--out'),{'status':'PQR_BINDING_COMPLETE_PENDING_PROBE','longest_request':{'id':'probe','split':'train'}})
  elif cmd=='pqr_runner.py':
   out=opt(args,'--out');out.mkdir(parents=True,exist_ok=True);split=args[args.index('--split')+1];bind=opt(args,'--binding');bsha=hashlib.sha256(bind.read_bytes()).hexdigest()
   if '--probe-id' in args:
    key=hashlib.sha256(b'probe').hexdigest();(out/(key+'.json')).write_text('{}');save_file({'corner_'+x:torch.zeros((1,49152),dtype=torch.bfloat16) for x in ('121','122','211','212')},str(out/(key+'.safetensors')));write(out/'_receipt.json',{'status':'PQR_PROBE_PENDING_REPLAY','split':split,'questions':1,'binding_sha256':bsha,'resources':{'newly_computed_questions':1,'measurement_scope':'model load through fresh full-question forward and saved-output validation; RSS whole process','max_new_question_bytes':100,'peak_reserved_bytes':3*c.GIB,'baseline_reserved_bytes':2*c.GIB,'peak_allocated_bytes':2*c.GIB,'baseline_allocated_bytes':c.GIB,'peak_rss_bytes':1000,'rss_bytes':900,'device_total_bytes':48*c.GIB,'pre_model_free_bytes':47*c.GIB}})
   elif split=='train' and state['pass']==0:
    (out/'uncommitted.fixture').write_text('fixture');raise RuntimeError('simulated interruption')
   else:write(out/'_receipt.json',{'status':'PQR_SPLIT_PENDING_COVERAGE','questions':{'train':1119,'validation':299,'test':1172}[split],'binding_sha256':bsha})
  elif cmd=='pqr_coverage.py':
   split=args[args.index('--split')+1];probe='--probe-id' in args;n=1 if probe else {'train':1119,'validation':299,'test':1172}[split];shards=opt(args,'--shards');key=hashlib.sha256(b'probe').hexdigest();q={'id':'probe','metadata_sha256':hashlib.sha256((shards/(key+'.json')).read_bytes()).hexdigest(),'logits_sha256':hashlib.sha256((shards/(key+'.safetensors')).read_bytes()).hexdigest()} if probe else {'id':'synthetic'};write(opt(args,'--report'),{'status':'PQR_LONGEST_PROBE_CUDA_REPLAY_ZERO' if probe else 'PQR_COMPLETE_CUDA_REPLAY_ZERO','binding_sha256':hashlib.sha256(opt(args,'--binding').read_bytes()).hexdigest(),'receipt_sha256':hashlib.sha256((shards/'_receipt.json').read_bytes()).hexdigest(),'split':split,'expected_questions':n,'complete_questions':n,'replayed_condition_logits':4*n,'questions':[q]})
  elif cmd=='pqr_analysis.py':write(opt(args,'--out'),{'status':'PQR_FULL_COVERAGE_DESCRIPTIVE_PENDING_REVIEW','splits':{s:{'all':{'n':n}} for s,n in [('train',1119),('validation',299),('test',1172)]}})
  else:raise AssertionError(cmd)
 c.run_owned=fake
 def setup():c.attempt=uuid.uuid4().hex;c.state={'status':'INITIALIZING','kind':'ROUND3_PQR_DRAFT_NOT_MECHANISM','attempt':c.attempt,'controller_sha256':'fixture'}
 setup();c.main();first=json.loads((c.run/'status.json').read_text());assert first['status']=='PQR_PROBE_COMPLETE_PENDING_ASTRA_REVIEW' and not any(cmd=='pqr_runner.py' and '--probe-id' not in args for _,cmd,args in state['calls'])
 review=c.run/'astra_review.json';write(review,{'status':'PQR_PROBE_REVIEW_APPROVED','reviewer':'Astra','binding_sha256':first['binding_sha256'],'probe_receipt_path':first['probe_receipt_path'],'probe_receipt_sha256':first['probe_receipt_sha256'],'probe_coverage_path':first['probe_coverage_path'],'probe_coverage_sha256':first['probe_coverage_sha256']})
 setup()
 try:c.main(str(review))
 except RuntimeError as e:assert 'simulated interruption' in str(e)
 else:raise AssertionError('interruption not propagated')
 assert json.loads((c.run/'status.json').read_text())['status']=='FAILED'
 state['pass']=1;setup();c.main(str(review));final=json.loads((c.run/'status.json').read_text());assert final['status']=='PQR_COMPLETE_PENDING_ASTRA_REVIEW';assert sum(cmd=='pqr_runner.py' and '--probe-id' in args for _,cmd,args in state['calls'])==3;assert len({str(opt(args,'--out')) for _,cmd,args in state['calls'] if cmd=='pqr_runner.py' and '--probe-id' in args})==3;assert len(list(c.run.glob('probe_coverage_*.json')))==3 and len(list(c.run.glob('prior_status_*.json')))==2
 for split in ('train','validation','test'):assert (c.run/('coverage_'+split+'.json')).exists()
with tempfile.TemporaryDirectory(dir=P/'research_plans/pqr_eight_corner_code_20261111') as td:run_test(pathlib.Path(td))
print('PQR_CONTROLLER_STAGE_RESUME_CPU_MOCK_PASS; no actual GPU replay')
