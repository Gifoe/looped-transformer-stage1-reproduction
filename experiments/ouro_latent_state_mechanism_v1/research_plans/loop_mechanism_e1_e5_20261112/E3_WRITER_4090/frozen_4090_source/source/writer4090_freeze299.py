"""Audit all 299 site references and freeze per-site donors before holdout effects."""
import json,hashlib,pathlib,torch
from safetensors.torch import load_file
import unrelated_reference_ledger as ledger
ROOT=pathlib.Path('/root/autodl-tmp/ouro-e1-e3-offload-20261113');Q=ROOT/'results';TRAIN=Q/'reference_train64';H=Q/'reference_validation299';L=Q/'frozen_unrelated_4090.json';SEL=ROOT/'inputs/WRITER_HOLDOUT_IDS_1030_299.json';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def load(base,c):
 stem=hashlib.sha256(c['id'].encode()).hexdigest();mp=base/(stem+'.json');tp=base/(stem+'.safetensors');assert sha(mp)==c['metadata_sha256'] and sha(tp)==c['tensor_sha256'];m=json.loads(mp.read_text());t=load_file(str(tp));assert m['id']==c['id'] and sha(tp)==m['tensor_sha256']
 for j in range(8,16):
  for k in ('a','b'):
   tag=f'layer{j:02d}_{k}';assert torch.equal(t[tag+'_N'].double()-t[tag+'_P'].double(),t[tag+'_delta']) and torch.isfinite(t[tag+'_delta']).all()
 return {'id':m['id'],'input_ids_sha256':m['input_ids_sha256'],'deltas':{(j,k):t[f'layer{j:02d}_{k}_delta'] for j in range(8,16) for k in ('a','b')}}
helper0=sha(pathlib.Path(ledger.__file__));assert sha(SEL)=='e7f9735f8ea6054e95fc82442953d5f591478cbb7cc137d08901fb70ed16a73a';tr=json.loads((TRAIN/'RECEIPT.json').read_text());hr=json.loads((H/'RECEIPT.json').read_text());frozen=json.loads(L.read_text());locked=json.loads(SEL.read_text());assert hr['status']=='4090_FRESH_VALIDATION299_REFERENCE_PENDING_INDEPENDENT_AUDIT' and [c['id'] for c in hr['cases']]==[c['id'] for c in locked['validation_299']] and len(hr['cases'])==299 and [c['id'] for c in tr['cases']]==frozen['pool_ids'] and all(sha(pathlib.Path(k))==v for k,v in hr['source_sha256'].items());pool=[load(TRAIN,c) for c in tr['cases']];out=Q/'frozen_unrelated_validation299.jsonl';assert not out.exists()
with out.open('x') as w:
 for i,c in enumerate(hr['cases']):
  r=load(H,c);mapping,_=ledger.unrelated_directions(r,pool);w.write(json.dumps({'id':r['id'],'input_ids_sha256':r['input_ids_sha256'],'sites':mapping},sort_keys=True)+'\n')
assert sha(pathlib.Path(ledger.__file__))==helper0 and sha(SEL)=='e7f9735f8ea6054e95fc82442953d5f591478cbb7cc137d08901fb70ed16a73a';receipt=Q/'frozen_unrelated_validation299_receipt.json';assert not receipt.exists();receipt.write_text(json.dumps({'status':'4090_VALIDATION299_REFERENCE_AUDITED_DONORS_FROZEN_BEFORE_TREATMENT','count':299,'reference_receipt_sha256':sha(H/'RECEIPT.json'),'pool_ledger_sha256':sha(L),'holdout_ids_sha256':sha(SEL),'mapping_sha256':sha(out),'helper_sha256':sha(pathlib.Path(ledger.__file__)),'script_sha256':sha(pathlib.Path(__file__)),'scope':'CPU audit actual site references and label-blind donor mapping; no treatment effect read.'},indent=2)+'\n');print('HOLDOUT_DONORS_FROZEN',sha(receipt),flush=True)
