"""Audit fresh reference tensors and freeze local per-site donors before any treatment."""
import json,hashlib,pathlib,torch
from safetensors.torch import load_file
import unrelated_reference_ledger as ledger
ROOT=pathlib.Path('/root/autodl-tmp/ouro-e1-e3-offload-20261113');Q=ROOT/'results';TRAIN=Q/'reference_train64';TECH=Q/'technical_reference_three';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def readcase(base,case):
 stem=hashlib.sha256(case['id'].encode()).hexdigest();mp=base/(stem+'.json');tp=base/(stem+'.safetensors');assert sha(mp)==case['metadata_sha256'] and sha(tp)==case['tensor_sha256'];m=json.loads(mp.read_text());t=load_file(str(tp));assert m['id']==case['id'] and m['tensor_sha256']==sha(tp)
 for j in range(8,16):
  for k in ('a','b'):
   tag=f'layer{j:02d}_{k}';n,p,d=[t[tag+'_'+x] for x in ('N','P','delta')];assert n.shape==p.shape==d.shape==(1,2048) and n.dtype==p.dtype==torch.bfloat16 and d.dtype==torch.float64 and torch.isfinite(d).all() and torch.equal(n.double()-p.double(),d)
 return {'id':m['id'],'input_ids_sha256':m['input_ids_sha256'],'deltas':{(j,k):t[f'layer{j:02d}_{k}_delta'] for j in range(8,16) for k in ('a','b')}}
def main():
 tr=json.loads((TRAIN/'RECEIPT.json').read_text());te=json.loads((TECH/'RECEIPT.json').read_text());assert tr['status']=='4090_FRESH64_REFERENCE_PENDING_INDEPENDENT_AUDIT' and te['status']=='4090_FRESH_THREE_REFERENCE_BARE_PASSED' and len(tr['cases'])==64 and len({x['id'] for x in tr['cases']})==64 and len(te['cases'])==3;assert all(sha(pathlib.Path(k))==v for k,v in tr['source_sha256'].items()) and all(sha(pathlib.Path(k))==v for k,v in te['source_sha256'].items());pool=[readcase(TRAIN,c) for c in tr['cases']];technical=[readcase(TECH,c) for c in te['cases']];mapping={}
 for r in pool+technical:
  m,_=ledger.unrelated_directions(r,pool);mapping[r['id']]={'input_ids_sha256':r['input_ids_sha256'],'sites':m}
 out=Q/'frozen_unrelated_4090.json';assert not out.exists();out.write_text(json.dumps({'status':'4090_UNRELATED_SITE_LEDGER_FROZEN_BEFORE_TREATMENTS','train_reference_receipt_sha256':sha(TRAIN/'RECEIPT.json'),'technical_receipt_sha256':sha(TECH/'RECEIPT.json'),'helper_sha256':sha(pathlib.Path(ledger.__file__)),'script_sha256':sha(pathlib.Path(__file__)),'pool_ids':[x['id'] for x in pool],'technical_ids':[x['id'] for x in technical],'mapping':mapping,'scope':'CPU delta/source audit and deterministic donor mapping only; model replay and treatment have not been executed.'},sort_keys=True,indent=2)+'\n');print('4090_FROZEN_DONORS',sha(out),flush=True)
if __name__=='__main__':main()
