"""Three-case technical P/Q/R attention interaction accounting from actual saved Q_R/K/V."""
import json,gzip,hashlib,time,os,datetime
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import numpy as np,torch
from safetensors.torch import load_file,save_file
from e1_fp64_attention import attention_row
from e3_value_partition import split_value_interaction
R=Path('/home/luansiyu/loop-latent'); B=R/'repo/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112';P=B/'E1/technical_probe';O=B/'E3/technical_partition';LED=R/'research_plans/prompt_boundary_audit_20261111/boundary_ledger.jsonl';COV=R/'research_plans/prefix_question_partition_draft_20261111/saved_offset_coverage.json';REF=R/'research_plans/natural_prefix_identity_20261111/reference_prefix_ids.json';CAT=B/'results/compact_input_catalog.jsonl.gz'
if not __debug__:raise RuntimeError('optimized Python forbidden')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
start=time.monotonic();lc=sha(LED);assert lc=='a77258ade36415efb17ca09a886417f9b265ce53feec1589431dff6062f4e2bc' and sha(COV)=='5263cc8bf07900822b11954194b7e9fd60c3066c9f43d96baee092157445dba6' and sha(REF)=='e984a118a098923ba87fab4ddd4a26d1f9da8c5dc9767c5ab073352cd3dfbf67' and sha(CAT)=='a38218bfe0c9940f326663b8ecc0447f875a18c07f0091813ddfef4a3d12342d';saved=json.loads(COV.read_text());assert saved['status']=='ALL_SAVED_OFFSET_PARTITIONS_PASS_INPUTS_ONLY' and saved['boundary_ledger_sha256']==lc;ref=json.loads(REF.read_text())['input_ids'];assert len(ref)==1839
with LED.open() as f:ledger={(r['split'],r['id']):r for l in f if (r:=json.loads(l))['split']=='test'}
with gzip.open(CAT,'rt') as f:cats={(r['split'],r['id']):r for l in f if (r:=json.loads(l))['split']=='test'}
pending=json.loads((P/'pending.json').read_text());replay=json.loads((P/'replay.json').read_text());assert replay['status']=='E1_THREE_CASE_CUDA_SCORE_AND_ENDPOINT_REPLAY_PASSED' and replay['pending_sha256']==sha(P/'pending.json')
O.mkdir(exist_ok=False); allout={};summ=[]
for i in range(3):
 mp=P/(str(i)+'.json');tp=mp.with_suffix('.safetensors');assert sha(mp)==pending['questions'][i]['metadata_sha256'] and sha(tp)==pending['questions'][i]['tensor_sha256'];m=json.loads(mp.read_text());t=load_file(str(tp));r=ledger[('test',m['id'])];c=cats[('test',m['id'])];ids=c['input_ids'];Rpos=c['R'];assert r['input_sha256']==c['input_ids_sha256']==m['input_ids_sha256'] and r['input_length']==len(ids) and r['readout_position']==Rpos==len(ids)-1 and r['groups']['prefix']==list(range(1839)) and r['groups']['current']==list(range(1839,len(ids))) and not r['crossing_offsets'] and not r['groups']['crossing'] and ids[:1839]==ref
 per={'id':m['id'],'input_sha256':m['input_ids_sha256'],'R':Rpos,'L':len(ids),'source_metadata_sha256':sha(mp),'source_tensor_sha256':sha(tp),'q':{}}
 for qi in (1,2):
  qR=t[f'natural_u{qi-1}_q'].float().numpy().astype('float64');k1=t['natural_u0_k'].float().numpy().astype('float64');k2=t['natural_u1_k'].float().numpy().astype('float64');v1=t['natural_u0_v'].float().numpy().astype('float64');v2=t['natural_u1_v'].float().numpy().astype('float64');assert qR.ndim==3 and k1.ndim==4 and k1.shape==k2.shape==v1.shape==v2.shape and k1.shape[1]==qR.shape[1] and k1.shape[-2]==len(ids);qfull=np.zeros((qR.shape[0],qR.shape[1],len(ids),qR.shape[-1]),dtype='float64');qfull[:,:,Rpos,:]=qR;kw=m['natural_capture'][str(qi)]['qkv']['kwargs'];assert kw['attn_mask'] is None and kw['is_causal'] is True and kw['dropout_p']==0
  A1,_,vld1=attention_row(qfull,k1,v1,Rpos,is_causal=True,scale=kw['scale'],enable_gqa=kw['enable_gqa']);A2,_,vld2=attention_row(qfull,k2,v1,Rpos,is_causal=True,scale=kw['scale'],enable_gqa=kw['enable_gqa']);assert vld1.all() and vld2.all() and np.allclose(A1,t[f'q{qi}_A1_fp64'].numpy(),rtol=1e-12,atol=1e-12) and np.allclose(A2,t[f'q{qi}_A2_fp64'].numpy(),rtol=1e-12,atol=1e-12)
  parts,total=split_value_interaction(A1,A2,v1,v2,Rpos,1839);ideal=t[f'q{qi}_interaction_fp64'].numpy();delta=t[f'q{qi}_delta_fp64'].numpy();assert np.allclose(total,ideal,rtol=1e-10,atol=1e-10);residual=delta-total
  for name,v in parts.items():allout[f'case{i}_q{qi}_{name}']=torch.from_numpy(v.copy())
  allout[f'case{i}_q{qi}_residual']=torch.from_numpy(residual.copy());per['q'][str(qi)]={'partition_max_abs':float(np.max(np.abs(parts['P']+parts['Q']+parts['R']-total))),'ideal_rebuild_max_abs':float(np.max(np.abs(total-ideal))),'scope':'test case for technical accounting only; not a scientific sample'}
 summ.append(per)
assert sha(LED)==lc and sha(REF)=='e984a118a098923ba87fab4ddd4a26d1f9da8c5dc9767c5ab073352cd3dfbf67' and sha(CAT)=='a38218bfe0c9940f326663b8ecc0447f875a18c07f0091813ddfef4a3d12342d';tp=O/'partition_components.safetensors';save_file(allout,str(tp));result={'status':'E3_THREE_CASE_CPU_PARTITION_TECHNICAL_PASS','questions':summ,'n':3,'elapsed_seconds':time.monotonic()-start,'output_sha256':sha(tp),'boundary_ledger_sha256':lc,'saved_offset_coverage_sha256':sha(COV),'prefix_reference_sha256':sha(REF),'catalog_sha256':sha(CAT),'prior_pending_sha256':sha(P/'pending.json'),'prior_replay_sha256':sha(P/'replay.json'),'script_sha256':sha(Path(__file__)),'scope':'No new GPU forward: reconstructed CPU attention rows from 3 actual previously captured Q_R/K/V, using global R and full original Q length; other Q rows are zeros solely to specify original shape. Per-position terms correspond to value-change positions, not key information origin. These test examples are technical only.'}
p=O/'receipt.json'
with p.open('x') as f:json.dump(result,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
print('E3_TECHNICAL_PARTITION_PASS',sha(p),flush=True)
