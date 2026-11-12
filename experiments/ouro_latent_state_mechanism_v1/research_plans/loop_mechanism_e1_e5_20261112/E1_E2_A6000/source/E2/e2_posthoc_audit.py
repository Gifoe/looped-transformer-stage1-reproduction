"""Post-hoc independent source and probe-state audit, without rewriting collection history."""
import json,gzip,hashlib,datetime,os
from pathlib import Path
import torch
from safetensors import safe_open
R=Path('/home/luansiyu/loop-latent');B=R/'repo/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112';E=B/'E2';BR=R/'research_plans/natural_branch_full_run_20261111';LC=R/'research_plans/NATURAL_L18_INDEPENDENT_COVERAGE_20261111.json';GEO=R/'research_plans/natural_r_geometry_20261111';CONFIG=R/'model/config.json';CAT=B/'results/compact_input_catalog.jsonl.gz'
if not __debug__:raise RuntimeError('optimized Python forbidden')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
r=json.loads((E/'E2_state_extract.receipt.json').read_text());rows=E/'E2_latent_transition_rows.jsonl.gz';states=E/'E2_fixed_probe_states.safetensors';source=GEO/'source.json';watch=[LC,source,CONFIG,CAT,rows,states];before={str(p):sha(p) for p in watch};assert before[str(LC)]=='da4478aef58b320f5aa079f85b06ac70eab524922c340b2326a9487020da2535'==r['l18_coverage_sha256'] and before[str(source)]=='8e22e6ef7d5cd920a7f5ac4faae0d7df31f002a094346e289cb325f9d5443ed2' and before[str(CONFIG)]=='ce9cc13da41591b8b4deca053d7dfee06424c0228628ee862ea86d725bc163f3' and before[str(CAT)]==r['catalog_sha256'] and before[str(rows)]==r['rows_sha256'] and before[str(states)]==r['probe_states_sha256'];gs=json.loads(source.read_text());links={(x['split'],x['id']):x for x in json.loads(LC.read_text())['questions']}
with gzip.open(rows,'rt') as f:rr=[json.loads(x) for x in f]
assert len(rr)==len(links)==2590 and {(x['split'],x['id']) for x in rr}==set(links)
with safe_open(str(states),framework='pt',device='cpu') as store:
 assert len(store.keys())==2590*4
 for split in ('train','validation','test'):
  gp=GEO/(split+'_rows.jsonl'); gh=sha(gp);assert gh==gs[split]['rows_sha256']; watch.append(gp);before[str(gp)]=gh;geo={}
  with gp.open() as f:
   for line in f:
    g=json.loads(line);assert g['id'] not in geo;geo[g['id']]=g
  for x in (x for x in rr if x['split']==split):
   ident=x['id'];g=geo[ident];pred=[max(range(len(s)),key=lambda j:s[j]) for s in x['native_scores'][:2]];correct=(pred[0]==x['gold'],pred[1]==x['gold']);group={(False,True):'rescue',(True,False):'damage',(True,True):'stable_correct',(False,False):'stable_wrong'}[correct];ties=any(sum(v==max(s) for v in s)>1 for s in x['native_scores'][:2]);assert x['group']==g['group']==group and x['natural_no_top_tie']==g['natural_no_top_tie']==(not ties) and x['demo']==g['demo'];link=links[(split,ident)];key=split+'_'+hashlib.sha256(ident.encode()).hexdigest();bp=BR/(key+'.safetensors');assert sha(bp)==link['branch_tensor_sha256']==x['branch_tensor_sha256']
   with safe_open(str(bp),framework='pt',device='cpu') as f:
    for u in (0,1):
     for dest,src in [(f'{key}_u{u}_l18_input',f'u{u}_layer17_x'),(f'{key}_u{u}_post_norm',f'loop{u+1}_post_norm')]:
      a=store.get_tensor(dest);b=f.get_tensor(src);assert a.dtype==b.dtype==torch.bfloat16 and torch.isfinite(a).all() and torch.isfinite(b).all() and torch.equal(a,b)
assert all(sha(Path(k))==v for k,v in before.items());result={'status':'E2_POSTHOC_SOURCE_AND_ALL_PROBE_STATES_AUDIT_PASSED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'count':len(rr),'probe_tensor_count':len(rr)*4,'source_sha256':before,'script_sha256':sha(Path(__file__)),'scope':'Independent post-hoc reread verifies current frozen sources, geometry labels/ties, and all four probe tensors per question against hashed original branch. It cannot prove the original collector checked these hashes before its execution; no such historical claim is made.'};p=E/'E2_posthoc_audit.json'
with p.open('x') as f:json.dump(result,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
print('E2_POSTHOC_AUDIT_COMPLETE',sha(p),flush=True)
