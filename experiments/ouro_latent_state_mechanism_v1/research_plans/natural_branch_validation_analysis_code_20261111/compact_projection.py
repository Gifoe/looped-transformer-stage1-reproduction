"""Per-question projection, directly derived from full independently-audited rows."""
import json,gzip,hashlib
from pathlib import Path
P=Path('/home/luansiyu/loop-latent/research_plans/natural_branch_validation_exploration_20261111')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rows=json.loads((P/'validation_rows.json').read_text())
out=P/'validation_projection.jsonl.gz'
with gzip.open(out,'xt') as f:
 for r in rows:
  outrow={k:r[k] for k in ('id','demo','group','natural_no_top_tie','gold','fixed_wrong','targets','natural_scores','telescoping')}
  outrow['layers']=[{'layer':x['layer'],**{branch:x[branch+'_delta'] for branch in ('attention','mlp')}} for x in r['layers']]
  f.write(json.dumps(outrow,sort_keys=True,allow_nan=False)+'\n')
metadata={'status':'projection_only_all_299_no_candidate_vectors','source_rows_sha256':sha(P/'validation_rows.json'),'projection_sha256':sha(out),'script_sha256':sha(Path(__file__)),'mapping':'retain row identity, group, targets, natural scores and telescoping; each layer retains attention/mlp delta (k1,k2,difference for allwrong/fixedwrong); omit candidate vectors and numerator/denominator parts'}
with (P/'validation_projection_receipt.json').open('x') as f:json.dump(metadata,f,sort_keys=True);f.write('\n')
print(metadata,flush=True)
