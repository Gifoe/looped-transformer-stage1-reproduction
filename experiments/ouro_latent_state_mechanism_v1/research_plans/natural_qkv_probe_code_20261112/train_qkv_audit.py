"""Train only CPU audit and score export; independent CUDA replay remains pending."""
import os,json,hashlib,gzip,math
from pathlib import Path
import torch
from safetensors.torch import load_file
import full_qkv,full_input,natural_sources,qkv_runner
from source_input import ROOT,sha
from state_capture import bits
RUN=full_qkv.OUT
AMEND=ROOT/'research_plans/NATURAL_QKV_DISCLOSURE_ORDER_20261112.md'
OUT=ROOT/'research_plans/natural_qkv_train_preliminary_20261112'
NAMES=full_qkv.NAMES
def require(v,msg):
 if not v:raise ValueError(msg)
def thash(t):return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
def main():
 require(sha(AMEND)=='5884b2acf145756b8d17ac7dfb1eb6aa34a227c28f567ab383e18359b8522d23','amendment');qs=full_input.coverage('train');keys={'train_'+hashlib.sha256(q['id'].encode()).hexdigest() for q in qs};require(len(keys)==1119 and keys=={p.stem for p in RUN.glob('train_*.json')}=={p.stem for p in RUN.glob('train_*.safetensors')},'cohort files');OUT.mkdir(exist_ok=False);rows=[];ledger=[]
 pr=ROOT/'research_plans/natural_qkv_probe_20261112';require(sha(pr/'receipt.json')=='7ecbfef7e735bc34c27c79393e538f2e01bff52f6fcdedb581444618b4e96acc' and sha(pr/'probe_pending.json')=='2c427b5813249016da0dec44edff795d563c2749dee26bb035c46ebb85c66a68' and sha(qkv_runner.LCOV)==qkv_runner.LCOV_SHA,'probe chain');pending=json.loads((pr/'probe_pending.json').read_text());mods=(full_qkv.qkv_observer,full_qkv.qkv_runner,full_qkv.replay_qkv,full_qkv.natural_l18_hook,full_qkv.natural_sources,full_qkv.technical,full_qkv.full_input,full_qkv.state_capture,full_qkv.source_input);require(all(sha(Path(m.__file__))==pending['source_hashes']['code_'+m.__name__] for m in mods),'actual module');geo=ROOT/'research_plans/natural_r_geometry_20261111';require(sha(geo/'source.json')=='8e22e6ef7d5cd920a7f5ac4faae0d7df31f002a094346e289cb325f9d5443ed2','geometry source');gs=json.loads((geo/'source.json').read_text());require(sha(geo/'train_rows.jsonl')==gs['train']['rows_sha256'],'geometry rows');geom={}
 with (geo/'train_rows.jsonl').open() as f:
  for line in f:
   v=json.loads(line);require(v['id'] not in geom,'geometry duplicate');geom[v['id']]=v
 require(set(geom)=={q['id'] for q in qs} and sum(x['demo'] for x in geom.values())==25,'geometry cohort')
 for q in qs:
  ident=q['id'];key='train_'+hashlib.sha256(ident.encode()).hexdigest();mp=RUN/(key+'.json');tp=mp.with_suffix('.safetensors');m=json.loads(mp.read_text());t=load_file(str(tp));src=natural_sources.load('train',ident);e=src['natural'];lp=ROOT/'research_plans/natural_l18_full_run_20261111';l18=load_file(str(lp/(key+'.safetensors')))
  require(m['id']==ident and m['split']=='train' and m['tensor_sha256']==sha(tp) and m['targets']==e['targets'] and m['gold']==e['gold'] and m['R']==e['R'] and m['input_ids_sha256']==hashlib.sha256(json.dumps(e['ids']).encode()).hexdigest() and m['natural_metadata_sha256']==e['metadata_sha256'] and m['natural_tensor_sha256']==e['tensor_sha256'] and m['branch_metadata_sha256']==src['source_metadata_sha256'] and m['branch_tensor_sha256']==src['source_tensor_sha256'] and m['l18_coverage_sha256']==sha(qkv_runner.LCOV) and m['l18_metadata_sha256']==sha(lp/(key+'.json')) and m['l18_tensor_sha256']==sha(lp/(key+'.safetensors')),'identity')
  require(all(m['code_hashes'][n]==pending['source_hashes']['code_'+n] for n in ('qkv_observer','qkv_runner','natural_sources','technical','full_input','state_capture','source_input','natural_l18_hook','replay_qkv')) and m['code_hashes']['full']==sha(Path(full_qkv.__file__)),'code')
  g=geom[ident];require(g['metadata_sha256']==e['metadata_sha256'] and g['tensor_sha256']==e['tensor_sha256'] and g['scores']==e['scores'][:2] and g['targets']==e['targets'] and g['gold']==e['gold'],'geometry binding');
  for depth in (1,2):
   cr=m['capture_receipts'][str(depth)];require(cr['status']=='SEALED_COMPLETE_NATURAL_FORWARD' and cr['id']==ident and cr['split']=='train' and cr['R']==e['R'] and cr['input_ids_sha256']==m['input_ids_sha256'] and cr['depth']==depth and cr['ut']==depth-1 and cr['physical_layer']==17 and cr['x_R_sha256']==thash(src['states'][depth-1]['x']) and cr['a_R_sha256']==thash(src['states'][depth-1]['a']),'capture')
  require(set(m['scores'])==NAMES and set(t)=={f'{n}_{s}' for n in NAMES for s in ('logits','sdpa_R','a_R')},'keys');receiver=m['capture_receipts']['1']['qkv']
  for n in NAMES:
   cr=m['corner_receipts'][n];shape=cr['actual_receiver_qkv']['qkv']['q']['shape'];z=t[n+'_logits'];sd=t[n+'_sdpa_R'];a=t[n+'_a_R'];require(cr['actual_receiver_qkv']==receiver and cr['root_input_events']==1 and cr['outside_R_unchanged'] is True and cr['receiver_x_R_sha256']==m['capture_receipts']['1']['x_R_sha256'] and len(shape)==4 and sd.dtype==torch.bfloat16 and list(sd.shape)==[shape[0],shape[1],shape[3]] and a.dtype==torch.bfloat16 and tuple(a.shape)==(1,2048) and z.dtype==torch.bfloat16 and tuple(z.shape)==(1,49152) and all(torch.isfinite(x).all() for x in (z,sd,a)) and thash(sd)==cr['injected_sdpa_R_sha256'] and thash(a)==cr['a_R_sha256'] and len(m['scores'][n])==len(e['targets']) and all(math.isfinite(v) for v in m['scores'][n]),'corner')
  require(bits(t['F11_logits'],e['logits'][0]) and bits(t['F22_logits'],l18['T12']) and bits(t['F11_a_R'],src['states'][0]['a']) and bits(t['F22_a_R'],src['states'][1]['a']) and m['scores']['F11']==e['scores'][0],'endpoint');scores=m['scores'];lm=json.loads((lp/(key+'.json')).read_text());require(scores['F22']==lm['scores']['T12'] and m['fixed_wrong']==lm['fixed_wrong']==max((j for j in range(len(e['targets'])) if j!=e['gold']),key=lambda j:e['scores'][0][j]),'L18 scores/fixed wrong');pred={k:max(range(len(v)),key=lambda j:v[j]) for k,v in scores.items()};ties={k:sum(x==max(v) for x in v)>1 for k,v in scores.items()};require(pred==m['predictions'] and ties==m['top_ties'],'behavior')
  target_logits={n:[float(t[n+'_logits'][0,i]) for i in e['targets']] for n in NAMES};rows.append({'id':ident,'split':'train','gold':e['gold'],'targets':e['targets'],'fixed_wrong':m['fixed_wrong'],'scores':scores,'natural_scores':e['scores'][:2],'group':g['group'],'demo':g['demo'],'natural_no_top_tie':g['natural_no_top_tie'],'predictions':pred,'top_ties':ties,'target_logits':target_logits,'metadata_sha256':sha(mp),'tensor_sha256':sha(tp)});ledger.append({'id':ident,'metadata_sha256':sha(mp),'tensor_sha256':sha(tp),'natural_metadata_sha256':e['metadata_sha256'],'natural_tensor_sha256':e['tensor_sha256'],'l18_metadata_sha256':m['l18_metadata_sha256'],'l18_tensor_sha256':m['l18_tensor_sha256']})
  if len(rows)%100==0:print('TRAIN_AUDITED',len(rows),flush=True)
 with gzip.open(OUT/'rows.jsonl.gz','xt',encoding='utf-8') as f:
  for row in rows:f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
 require(sha(geo/'train_rows.jsonl')==gs['train']['rows_sha256'],'geometry drift');out={'geometry_source_sha256':sha(geo/'source.json'),'geometry_rows_sha256':sha(geo/'train_rows.jsonl'),'status':'CPU_SOURCE_ENDPOINT_PASS_INDEPENDENT_CUDA_PENDING','count':len(rows),'questions':ledger,'rows_sha256':sha(OUT/'rows.jsonl.gz'),'amendment_sha256':sha(AMEND),'script_sha256':sha(Path(__file__)),'limitation':'collector CUDA score readback is not an independent replay; score values retained without CPU replacement'}
 with (OUT/'coverage.json').open('x') as f:json.dump(out,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
 print('QKV_TRAIN_CPU_AUDIT_COMPLETE',sha(OUT/'coverage.json'),flush=True)
if __name__=='__main__':main()
