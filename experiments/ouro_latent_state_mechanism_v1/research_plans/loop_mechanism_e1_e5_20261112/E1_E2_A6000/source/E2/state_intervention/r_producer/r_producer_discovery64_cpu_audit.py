"""Independent CPU provenance and target17 R/a anchor check for R-producer64."""
import json,gzip,pathlib,hashlib,torch
from safetensors.torch import load_file
from safetensors import safe_open
R=pathlib.Path('/home/luansiyu/loop-latent');B=R/'repo/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112';E=B/'E2';P=E/'state_intervention/r_producer/discovery64';PR=E/'state_intervention/discovery64';BR=R/'research_plans/natural_branch_full_run_20261111';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def bits(a,b):return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
sel=json.loads((B/'orders/E1_DISCOVERY64_FROZEN.json').read_text());ids={q['id'] for group in sel['groups'].values() for q in group};pending=json.loads((P/'PENDING.json').read_text());assert len(ids)==len(pending['cases'])==64 and {x['id'] for x in pending['cases']}==ids
with gzip.open(E/'E2_latent_transition_rows.jsonl.gz','rt') as f:rows={(r['split'],r['id']):r for r in map(json.loads,f)}
prior=json.loads((PR/'PENDING_INDEPENDENT_AUDIT.json').read_text());links={(x['split'],x['id']):x for x in prior['cases']};done=[]
for q in pending['cases']:
 key=hashlib.sha256((q['split']+'\0'+q['id']).encode()).hexdigest();mp=P/(key+'.json');tp=P/(key+'.safetensors');assert sha(mp)==q['metadata_sha256'] and sha(tp)==q['tensor_sha256'];m=json.loads(mp.read_text());t=load_file(str(tp));priorp=PR/(key+'.safetensors');priorm=PR/(key+'.json');link=links[(q['split'],q['id'])];assert sha(priorp)==m['prior_tensor_sha256']==link['tensor_sha256'] and sha(priorm)==m['prior_metadata_sha256']==link['metadata_sha256'];old=load_file(str(priorp));assert bits(t['target17_logits'],old['rollback_full_logits']) and bits(t['target17_x_R'],old['rollback_full_injected_R']) and bits(t['target17_a_R'],old['rollback_full_a_R']) and bits(t['native2_logits'],old['native2_logits']);rr=rows[(q['split'],q['id'])];bp=BR/(q['split']+'_'+hashlib.sha256(q['id'].encode()).hexdigest()+'.safetensors');assert sha(bp)==m['branch_sha256']==rr['branch_tensor_sha256']
 with safe_open(str(bp),framework='pt',device='cpu') as f:
  for cut in (0,8,16):
   a=f.get_tensor(f'u0_layer{cut:02d}_x');b=f.get_tensor(f'u1_layer{cut:02d}_x');assert bits(a,t[f'cut{cut}_cut_after_R']) and bits(b,t[f'cut{cut}_cut_original_R']);assert abs(float((a.double()-b.double()).norm())-m['cuts'][str(cut)]['injection_delta_l2'])<1e-10 and m['cuts'][str(cut)]['target_nonR_matches_clean'] and m['cuts'][str(cut)]['R_only']
 done.append(q['id'])
out=P/'CPU_SOURCE_ANCHOR_AUDIT.json';assert not out.exists();out.write_text(json.dumps({'status':'R_PRODUCER_64_CPU_SOURCE_AND_TARGET_ANCHOR_PASSED','n':len(done),'ids':done,'pending_sha256':sha(P/'PENDING.json'),'script_sha256':sha(pathlib.Path(__file__))},indent=2)+'\n');print('CPU_ANCHOR_PASS',sha(out))
