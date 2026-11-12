import json,gzip,hashlib,pathlib
b=pathlib.Path('/root/autodl-tmp/ouro-e1-e3-offload-20261113');cat=b/'inputs/compact_input_catalog.jsonl.gz';inp=pathlib.Path('/root/autodl-tmp/ouro-e4-e5/inputs/runner_inputs.jsonl');out=b/'inputs/INPUT_CATALOG_AUDIT.json';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert sha(cat)=='a38218bfe0c9940f326663b8ecc0447f875a18c07f0091813ddfef4a3d12342d';q={(x['split'],x['id']):x for line in inp.read_text().splitlines() if (x:=json.loads(line))};n=0
with gzip.open(cat,'rt') as f:
 for line in f:
  c=json.loads(line);x=q[(c['split'],c['id'])];assert x['ids']==c['input_ids'] and x['targets']==c['targets'] and x['R']==c['R'] and x['metadata_sha256']==c['metadata_sha256'];n+=1
assert n==len(q)==2590;result={'status':'ALL_2590_INPUT_IDS_TARGETS_R_METADATA_MATCH_A6000_CATALOG','count':n,'catalog_sha256':sha(cat),'runner_input_sha256':sha(inp),'scope':'CPU input identity only; no cross-hardware score equality asserted.'};out.write_text(json.dumps(result,indent=2)+'\n');print(sha(out))
