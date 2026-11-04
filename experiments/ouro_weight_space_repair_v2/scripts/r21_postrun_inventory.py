"""Post-run R21 inventory and certificate-range diagnostics; no new LP solve."""
import json,hashlib
from pathlib import Path
from collections import Counter
R=Path(__file__).resolve().parents[1]
def main():
 rows=[json.loads(x) for x in (R/'results/r21_oracle.jsonl').read_text().splitlines()];done=json.loads((R/'results/r21_oracle_COMPLETE.json').read_text());sup=json.loads((R/'logs/r21_SUPERVISOR.json').read_text());assert len(rows)==done['n']==576 and sup['status']=='complete' and not sup['overrun']
 certs=[c for r in rows for key in ['H6','H7'] for c in r[key]]
 out=dict(status='complete_post_run_inventory',n=576,total_LP_certificates=len(certs),invalid_certificates=sum(not c['certificate_valid'] for c in certs),all_label_status_counts={key:dict(Counter(c['status'] for r in rows for c in r[key])) for key in ['H6','H7']},reachable_label_count_histograms={key:dict(Counter(r[key+'_reachable_label_count'] for r in rows)) for key in ['H6','H7']},maximum_abs_raw_residual_or_gap=max(abs(v) for c in certs for v in c['raw_residuals'].values()),maximum_roundoff_delta=max(c['roundoff_delta'] for c in certs),raw_sha256=hashlib.sha256((R/'results/r21_oracle.jsonl').read_bytes()).hexdigest(),GPU_worker_seconds=0,CPU_wall_seconds=sup['CPU_wall_seconds'])
 with (R/'results/r21_POSTRUN_INVENTORY.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()
